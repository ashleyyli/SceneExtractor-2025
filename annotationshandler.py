import decord
import numpy as np
import cv2

def bgremove1(myimage):
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage,(5,5), 0)
 
    # We bin the pixels. Result will be a value 1..5
    bins=np.array([0,51,102,153,204,255])
    myimage[:,:,:] = np.digitize(myimage[:,:,:],bins,right=True)*51
 
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret,background = cv2.threshold(myimage_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
    # Convert mask to 3-channel image and set background to black
    background = np.zeros_like(myimage)
    background[:, :] = (0, 0, 0)  # Black background
    background[ret == 0] = (0, 0, 0)  # Ensure background is set

    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foreground
    ret,foreground = cv2.threshold(myimage_grey,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)  # Currently foreground is only a mask
    # Set original (0,0,0) pixels to (1,1,1) to avoid miscount in remaining pixels
    # TODO: doesn't seem to work
    myimage[myimage == 0] = 1
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground

    return finalimage


# TODO: function name is not descriptive
def compare_annotations_difference(curr, ref):
    """
    Compares the difference between two frames and determine if most of curr is 
        contained within the reference.
    
    Args:
        curr: (Chronologically earlier) Previous frame.
        ref: Reference frame.
    
    Returns:
        bool: True if most of curr is contained within ref, False otherwise.
    """

    # Calculate differences between frames at each pixel
    diff_squared = np.sum((ref - curr)**2, axis=2) 

    # Could put a filter on diff_squared to reduce weight of differences towards the edges of the frame

    # For each pair of pixels, if pixels are similar enough, then ref contains curr and declare no difference
    curr[diff_squared < 100] = 0

    # Count different pixels
    num_different = np.count_nonzero(curr)

    # Return true if sufficiently similar
    return num_different < len(curr) * 0.015 


def filter_annotations(video_path, frame_cuts):
    """
    Filters out annotations from frame cuts
    
    Args:
        video_path (str): Path to the video file.
        frame_cuts (list): List of frame numbers where cuts are predicted.
    
    Returns:
        list: Frame cuts without frames detected as differing only by annotations.
    """

    # Reverse frame_cuts to enumerate backwards
    frame_cuts = list(reversed(frame_cuts))

    # TODO: Decide whether to return new frames or modify array in place
    filtered_frame_cuts = [frame_cuts[0]] 

    # Load video reader
    vr_full = decord.VideoReader(video_path, ctx=decord.cpu(0))

    # Enumerate backwards through frame cuts, comparing against most recently added filtered_frame_cuts element, and check for annotations
    # If true (difference below threshold),  do not add to filtered_frame_cuts; else add 
    frame_vr = vr_full[filtered_frame_cuts[0]]
    reference_frame = cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2BGR)
    processed_reference_frame = bgremove1(reference_frame)

    for i in range(1, len(frame_cuts)):
        frame_vr = vr_full[frame_cuts[i]]
        curr_frame = cv2.cvtColor(frame_vr.asnumpy(), cv2.COLOR_RGB2BGR)
        processed_curr_frame = bgremove1(curr_frame)

        # If current frame is contained within reference frame, skip
        if compare_annotations_difference(processed_curr_frame, processed_reference_frame):
            continue

        # Else, add current frame and update reference frame (processed_reference_frame)
        filtered_frame_cuts.append(frame_cuts[i])
        processed_reference_frame = processed_curr_frame

    return list(reversed(filtered_frame_cuts))  


