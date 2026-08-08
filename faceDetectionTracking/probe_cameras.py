import cv2

MAX_INDEX_TO_CHECK = 5  # bump this up if you have more devices than that


def probe():
    print("Camera index probe")
    print("-------------------")
    print("For each index found, a window will open showing that camera's feed.")
    print("Controls: press 'n' to check the NEXT index, 'q' to quit the probe.")
    print("Note down the index number that shows YOUR ACTUAL FACE, not the OBS logo.\n")

    for index in range(MAX_INDEX_TO_CHECK):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Index {index}: could not open (likely no device here) - skipping")
            cap.release()
            continue

        print(f"\nIndex {index}: opened. Showing preview window - check if this is your real camera.")
        print("(press 'n' for next index, 'q' to stop probing)")

        quit_requested = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Index {index}: opened but couldn't read frames - probably not usable")
                break

            # Overlay the index number directly on the frame so it's
            # unambiguous which index you're currently looking at.
            display = frame.copy()
            cv2.putText(display, f"Camera index: {index}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, "press 'n' = next, 'q' = quit", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Probe', display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                quit_requested = True
                break

        cap.release()
        cv2.destroyAllWindows()

        if quit_requested:
            break

    print("\nProbe finished. Use the index that showed your real face as")
    print("CAMERA_INDEX in main.py.")


if __name__ == "__main__":
    probe()