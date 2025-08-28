import os
import decord
import numpy as np
import cv2

def compare_scroll_difference(curr, ref, width, height):
    """
    Compares two frames are scrolled versions of each other
    
    Args:
        curr: (Chronologically earlier) Previous frame.
        ref: Reference frame.
        width: Width of the frames.
        height: Height of the frames.
    
    Returns:
        bool: True if most of curr is contained within next, False otherwise.
    """

    # All the 6 methods for comparison in a list
    # methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
    #             'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']

    method = getattr(cv2, 'TM_SQDIFF_NORMED')    

    # Defining region of search
    img_w = int(width/2)
    img_h = int(height/2)
    img_x = int(width/4)
    img_y = int(height/4)

    # create target (?????)
    crop_image = curr[img_y:img_y+img_h, img_x:img_x+img_w]

    # Apply template matching
    res = cv2.matchTemplate(crop_image, ref, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Printing
    print("Min: ", min_val, " ", min_loc)
    # print("Max: ", max_val, " ", max_loc)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + img_w, top_left[1] + img_h)           # how does python scope work???
    
    return min_val < 0.005


def find_match(curr, ref, width, height, index):
    # Defining region of search
    img_x = int(width/6)
    img_y = int(height/6)    
    img_w = int(2 * width/3)
    img_h = int(2 * height/3)


    # create target (?????)
    crop_image = ref[img_y:img_y+img_h, img_x:img_x+img_w]

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(crop_image, None)
    kp2, des2 = sift.detectAndCompute(curr, None)

    # If no features were found in either image, skip
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print(f"No descriptors found (des1={des1 is None}, des2={des2 is None}) at index {index}")
        return False

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # print("Num matches: ", len(matches))

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
    
    # print("Good matches: ", len(good))
    # if len(matches) > 0:
    #     print("Ratio: ", len(good)/len(matches))
    # else:
    #     print("Ratio: undefined")

    if len(good) < 100:
        print("Not enough good matches to estimate motion")
        return False

    # Calculating distances for matched pairs
    src_pts = np.float32([(kp1[m.queryIdx].pt[0] + img_x, kp1[m.queryIdx].pt[1] + img_y) for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    displacements = dst_pts - src_pts

    dx = displacements[:,0] / width
    dy = displacements[:,1] / height

    # print("dx mean and std dev: ", np.mean(dx), np.std(dx))
    # print("dy mean and std dev: ", np.mean(dy), np.std(dy))

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    # print("Num inliers:", np.sum(inliers))

    inlier_ratio = np.sum(inliers) / len(inliers)
    if inlier_ratio < 0.3:  # threshold depends on your data
        print("Reject: not enough inliers")
        return False

    tx = M[0,2]
    ty = M[1,2]
    sx = np.sqrt(M[0,0]**2 + M[1,0]**2)  # scale in x
    sy = np.sqrt(M[0,1]**2 + M[1,1]**2)  # scale in y

    # print ("tx:", tx, "ty:", ty, "sx:", sx, "sy:", sy)


    draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor = (0, 0, 255),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

    output = cv2.drawMatchesKnn(crop_image, kp1, curr, kp2, matches, None, **draw_params)

    # Output result
    DATA_DIR = os.getcwd()
    DATA_DIR_NAME = 'test'
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
    reference_frame_pc = np.float32(cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2GRAY))
    # scores = []

    index = 0
    for i in range(1, len(frame_cuts)):
        print("\n", i)
        print(f"Comparing frame {frame_cuts[i]} with reference frame {filtered_frame_cuts[-1]}")
        frame_vr = vr_full[frame_cuts[i]]
        curr_frame = cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2BGR)
        curr_frame_pc = np.float32(cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2GRAY))


        # If reference frame is contained within current frame, skip
        # if compare_scroll_difference(curr_frame, reference_frame, width, height):
        #     find_match(curr_frame, reference_frame, width, height, i)
        #     print(f"Frame {frame_cuts[i]} is a scrolling frame, skipping.")
        #     continue

        shift, response = cv2.phaseCorrelate(reference_frame_pc, curr_frame_pc)
        if (response > 0.40):
            dx, dy = shift
            if (abs(dy) > 1):
                print(f"Phase correlation: Frame {frame_cuts[i]} is a scrolling frame, skipping.")
                continue

        # If reference frame is reasonably found within current frame, skip
        # Covers some cases template matching does not, but very expensive
        if find_match(curr_frame, reference_frame, width, height, i): 
            print(f"Feature match: Frame {frame_cuts[i]} is a scrolling frame, skipping.")
            continue

        # Else, add current frame and update reference frame
        filtered_frame_cuts.append(frame_cuts[i])
        reference_frame = curr_frame
        reference_frame_pc = curr_frame_pc

    # print(scores)
    return list(reversed(filtered_frame_cuts))      
    # return list(reversed(frame_cuts))  