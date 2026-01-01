import os
import decord
import numpy as np
import cv2

def find_match(curr, ref, width, height, index):
    # Search for the middle of reference frame in the current frame
    # Ignore edges because they can cloud motion estimation
    img_x = int(width/6)
    img_y = int(height/6)
    img_w = int(2 * width/3)
    img_h = int(2 * height/3)

    # Create target
    crop_image = ref[img_y:img_y+img_h, img_x:img_x+img_w]

    # Initiate scale-invariant feature detector (Scale Invariant Feature 
    # Transform) from Lowe's paper
    # https://doi.org/10.1023/B:VISI.0000029664.99615.94
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(crop_image, None)
    kp2, des2 = sift.detectAndCompute(curr, None)

    # If no features were found in either image, skip
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print(f"No descriptors found (crop_image={des1 is None}, curr={des2 is None}) at index {index}")
        return False

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1  # Use k-d trees for nearest neighbor search
    NUM_TREES = 5  # Default, more trees is faster but uses more memory
    NUM_CHECKS = 50  # Number of neighbors to check
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = NUM_TREES) 
    search_params = dict(checks = NUM_CHECKS)  

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # DEBUG: Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper (7.1 Keypoint matching)
    DISTANCE_RATIO = 0.7  
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < DISTANCE_RATIO * n.distance:
            good.append(m)

    print("Matches: ", len(matches))
    print("Good: ", len(good))
    print("Good ratio: ", len(good)/len(matches))

    # If fewer than MIN_GOOD found between frames, then it's unlikely
    # they are scrolled versions of each other
    MIN_GOOD_RATIO = 0.2  # Increase to require more matches for two frames to be considered related
    if len(good)/len(matches) < MIN_GOOD_RATIO:
        print("Not enough good matches to estimate motion")
        return False

    # Calculating distances for matched pairs
    src_pts = np.float32([(kp1[m.queryIdx].pt[0] + img_x, kp1[m.queryIdx].pt[1] + img_y) for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    # Fraction of matched features that agree with the estimated motion
    inlier_ratio = np.sum(inliers) / len(inliers)
    print("Inlier ratio: ", inlier_ratio)       # DEBUG: Print inlier ratio
    
    # If not enough agree, it's unlikely the frames are scrolled versions of each other
    MIN_INLIER_RATIO = 0.3 # Increase to require more consistent motion between frames 
    if inlier_ratio < MIN_INLIER_RATIO:
        print("Reject: not enough inliers")
        return False

    # Calculating translation and scale
    tx = M[0,2]  # translation in x
    ty = M[1,2]  # translation in Y
    # sx = np.sqrt(M[0,0]**2 + M[1,0]**2)  # scale in x
    sy = np.sqrt(M[0,1]**2 + M[1,1]**2)  # scale in y

    # DEBUG: printing translation and scale
    print("x-shift: ", tx)
    print("y-shift: ", ty)
    print("y-scale: ", sy)

    # DEBUG: Output result image
    draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = (0, 0, 255),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

    output = cv2.drawMatchesKnn(crop_image, kp1, curr, kp2, matches, None, **draw_params)

    DATA_DIR = os.getcwd()
    DATA_DIR_NAME = 'full_test_matches'
    out_directory = os.path.join(DATA_DIR, DATA_DIR_NAME)
    os.makedirs(out_directory, exist_ok=True)
    img_file = os.path.join(
        out_directory, f"matched_frame-{index}.jpg")
    cv2.imwrite(img_file, output)

    if abs(sy - 1) > 0.01 or abs(ty) > 1 or abs(tx) > 1:
        return True

    return False


def filter_scrolling(video_path, frame_cuts, width, height):
    """
    Filters out scrolling frames from frame cuts
    
    Args:
        video_path (str): Path to the video file.
        frame_cuts (list): List of frame numbers where cuts are predicted.
    
    Returns:
        list: Frame cuts without frames detected as differing only by scrolling.
    """

    # Reverse frame_cuts to enumerate backwards
    frame_cuts = list(reversed(frame_cuts))

    # TODO: Decide whether to return new frames or modify array in place
    filtered_frame_cuts = [frame_cuts[0]] 

    # Load video reader
    vr_full = decord.VideoReader(video_path, ctx=decord.cpu(0))

    # Enumerate backwards through frame cuts, comparing against most recently added filtered_frame_cuts element, and check for scrolling
    # If true (difference below threshold),  do not add to filtered_frame_cuts; else add 
    frame_vr = vr_full[filtered_frame_cuts[0]]
    reference_frame = cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2BGR)

    for i in range(1, len(frame_cuts)):
        print("\n", i)
        print(f"Comparing frame {frame_cuts[i]} with reference frame {filtered_frame_cuts[-1]}")
        frame_vr = vr_full[frame_cuts[i]]
        curr_frame = cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2BGR)

        # difference = abs(curr_frame - reference_frame)
        # Arbitrary threshold
        # if ((np.count_nonzero(difference) / curr_frame.size) < 0.05) :
        #     print("Duplicate frames")
        #     continue;

        # If reference frame is reasonably found within current frame, skip
        if find_match(curr_frame, reference_frame, width, height, i): 
            print(f"Feature match: Frame {frame_cuts[i]} is a scrolling frame, skipping.")
            continue

        # Else, add current frame and update reference frame
        filtered_frame_cuts.append(frame_cuts[i])
        reference_frame = curr_frame

    return list(reversed(filtered_frame_cuts))      
