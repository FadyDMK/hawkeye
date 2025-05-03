import cv2
def select_roi(image, max_disp=(960, 540)):
    # Show a scaled-down version of `image` for ROI selection so it fits on-screen,
    # then map the selected ROI back to original coordinates.
    h, w = image.shape[:2]
    max_w, max_h = max_disp
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.namedWindow("Select Court ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Court ROI", disp.shape[1], disp.shape[0])
    r = cv2.selectROI("Select Court ROI", disp, showCrosshair=True)
    cv2.destroyWindow("Select Court ROI")
    x, y, rw, rh = [int(v) for v in r]
    # map back
    x0, y0 = int(x / scale), int(y / scale)
    w0, h0 = int(rw / scale), int(rh / scale)
    if w0 == 0 or h0 == 0:
        return image, (0, 0)
    roi = image[y0:y0+h0, x0:x0+w0]
    return roi, (x0, y0)