import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from ocr_orientation import detect_orientation
import torch
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)


class ChessBoard:
    def __init__(self, model_path="weights/xcorners-yolo11n.pt"):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.centers = []
        self.quads = []
        self.quad_centers = []
        self.grouped_rows = []
        self.H = None
        self.orientation = "unknown"
        self.rotation = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the model
        if 'model' in state:
            del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-create the model
        self.model = YOLO(self.model_path)
        
    def find_board2(self, frame):
        for angle in [-99, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            frame_ = frame.copy()
            if angle == -99:
                pass
            else:
                frame_ = cv2.rotate(frame_, angle)
                self.rotation = angle
                
            if self.find_board(frame_):
                return True
            
        return False 
            
            

    def find_board(self, frame):
        """
        Main method to find the chessboard, estimate homography, and determine orientation.
        """
        # Try rotating if detection fails
        # for angle in [
        #     cv2.ROTATE_90_CLOCKWISE,
        #     cv2.ROTATE_180,
        #     cv2.ROTATE_90_COUNTERCLOCKWISE,
        # ]:
        # #     rotated_img = cv2.rotate(corner_cell_img, angle)
        # #     cell_id = detect_orientation(rotated_img)
        # #     if cell_id:
        # #         self.orientation = f"{cell_id}_{name}"
        # #         return
        self._detect_corners(frame)
        if len(self.centers) < 4:
            print("Not enough corners detected to form a board.")
            return False

        self._find_quads()
        self._group_quads_into_rows()

        if (
            not self.grouped_rows
            or sum(len(row) for row in self.grouped_rows) < 4
        ):
            print("Failed to group corners into rows.")
            return False

        self._estimate_homography()
        if self.H is None:
            print("Homography estimation failed.")
            return False
        self._determine_orientation(frame)
        if self.orientation == "unknown":
            print("Orientation determination failed.")
            return False
        return True

    def _detect_corners(self, frame):
        """
        Detects potential corners of chessboard squares using a YOLO model.
        """
        results = self.model.predict(frame, save=False, verbose=False, device=device)
        self.centers = []
        for x1, y1, x2, y2 in results[0].boxes.xyxy.cpu().numpy():
            xc = 0.5 * (x1 + x2)
            yc = 0.5 * (y1 + y2)
            self.centers.append([xc, yc])

    def _find_quads(self):
        """
        Finds quadrilaterals from the detected corners using Delaunay triangulation.
        """
        pts = np.asarray(self.centers, dtype=np.float32)
        if len(pts) < 3:
            return

        tri = Delaunay(pts)
        tri_longest_edge = {}
        edge_to_tris = {}

        for t_id, (i0, i1, i2) in enumerate(tri.simplices):
            p0, p1, p2 = pts[i0], pts[i1], pts[i2]
            u_loc, v_loc = self._longest_edge_local(p0, p1, p2)
            tri_local_to_global = [i0, i1, i2]
            gi, gj = (
                int(tri_local_to_global[u_loc]),
                int(tri_local_to_global[v_loc]),
            )
            edge = tuple(sorted((gi, gj)))
            tri_longest_edge[t_id] = edge
            edge_to_tris.setdefault(edge, []).append(t_id)

        quads_poly = []
        seen_keys = set()
        for edge, t_ids in edge_to_tris.items():
            if len(t_ids) == 2:
                t1, t2 = t_ids
                if (
                    tri_longest_edge.get(t1) == edge
                    and tri_longest_edge.get(t2) == edge
                ):
                    verts1 = set(tri.simplices[t1])
                    verts2 = set(tri.simplices[t2])
                    all_verts = verts1 | verts2
                    if len(all_verts) == 4:
                        key = tuple(sorted(int(v) for v in all_verts))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            poly = np.array(
                                [pts[i] for i in all_verts], dtype=np.float32
                            )
                            poly = poly[self._order_ccw(poly)]
                            quads_poly.append(poly)
        self.quads = quads_poly
        self.quad_centers = []
        for i, q in enumerate(self.quads):
            cx = np.mean(q[:, 0])
            cy = np.mean(q[:, 1])
            self.quad_centers.append((i, cx, cy))

    def _group_quads_into_rows(self):
        """
        Groups the found quadrilaterals into rows based on their y-coordinates.
        """
        if not self.quad_centers:
            self.grouped_rows = []
            return

        y_coords = np.array([c[2] for c in self.quad_centers]).reshape(-1, 1)
        db = DBSCAN(eps=20, min_samples=1).fit(y_coords)
        labels = db.labels_

        rows_dict = {}
        for i, center in enumerate(self.quad_centers):
            label = labels[i]
            rows_dict.setdefault(label, []).append(center)

        sorted_row_labels = sorted(
            rows_dict.keys(),
            key=lambda label: np.mean([c[2] for c in rows_dict[label]]),
        )

        grouped = []
        for label in sorted_row_labels:
            row = sorted(rows_dict[label], key=lambda c: c[1])
            grouped.append(row)
        self.grouped_rows = grouped

    def _estimate_homography(self):
        """
        Estimates the homography matrix from the grouped rows of quad centers.
        """
        rc_pts, img_pts = [], []
        for r, row in enumerate(self.grouped_rows, start=1):
            for c, item in enumerate(row, start=1):
                _, cx, cy = item
                rc_pts.append([r, c])
                img_pts.append([cx, cy])

        if len(rc_pts) < 4:
            self.H = None
            return

        rc_pts = np.asarray(rc_pts, np.float32)
        img_pts = np.asarray(img_pts, np.float32)
        xy = rc_pts[:, ::-1].copy()

        H, _ = cv2.findHomography(
            xy, img_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        self.H = H
        

        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            print("Error: Homography matrix is not invertible.")
            self.H_inv = None

    def _determine_orientation(self, frame):
        """
        Determines the board's orientation by checking the corners.
        """
        corners = {
            "bottom_left": (7, 0),
            "bottom_right": (7, 7),
            "top_left": (0, 0),
            "top_right": (0, 7),
        }
        for name, (row, col) in corners.items():
            uv = self.project_rc_center(row, col)
            u, v = int(uv[0]), int(uv[1])
            buffer = int(max(abs(self.H[0, 0]), abs(self.H[1, 1]), 20))
            corner_cell_img = frame[
                max(0, v - buffer) : v + buffer, max(0, u - buffer) : u + buffer
            ]
            if corner_cell_img.size == 0:
                continue
            gray = cv2.cvtColor(corner_cell_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cell_id = detect_orientation(gray)
            if cell_id:
                self.orientation = f"{cell_id}_{name}"
                return


    def project_rc_center(self, row, col):
        """Projects a single (row, col) to image coordinates."""
        if self.H is None:
            return None
        xy = np.array([[col, row]], np.float32).reshape(-1, 1, 2)
        uv = cv2.perspectiveTransform(xy, self.H).reshape(-1, 2)
        return uv[0]

    def project_xy_to_rc(self, x, y):
        """Projects a single (x, y) image coordinate to (row, col)."""
        if self.H_inv is None:
            return None


        uv = np.array([[x, y]], np.float32).reshape(-1, 1, 2)
        rc = cv2.perspectiveTransform(uv, self.H_inv).reshape(-1, 2)
        # The result is in (col, row) format, so we reverse it for clarity
        row, col = rc[0, 1], rc[0, 0]
        return int(round(row)), int(round(col))

    def get_pgn_from_rc(self, r, c, rows=8, cols=8):
        """Calculates PGN notation from row and column based on orientation."""
        fallback = (f"r{r}", f"c{c}")
        if self.orientation == "unknown":
            return fallback
        
        parts = self.orientation.split("_")
        text, corner_name = parts[0], "_".join(parts[1:])
        if len(text) != 2:
            return fallback

        file_char, rank_char = text[0], text[1]

        if "left" in corner_name:
            file_increasing = file_char == "a"
        elif "right" in corner_name:
            file_increasing = file_char == "h"
        else:
            return fallback

        if "top" in corner_name:
            rank_increasing = rank_char == "1"
        elif "bottom" in corner_name:
            rank_increasing = rank_char == "8"
        else:
            return fallback

        pgn_col = (
            chr(ord("a") + c)
            if file_increasing
            else chr(ord("a") + (cols - 1 - c))
        )
        pgn_row = str(1 + r) if rank_increasing else str(rows - r)
        return pgn_col, pgn_row

    def draw_projected_centers(self, frame, rows=8, cols=8, color=(0, 0, 255)):
        """Draws projected centers and their PGN labels on the frame."""
        if self.H is None:
            return
        uv = self._project_rc_centers(rows, cols)
        for r in range(rows):
            for c in range(cols):
                x, y = uv[r, c]
                pgn_col, pgn_row = self.get_pgn_from_rc(r, c, rows, cols)
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                cv2.putText(
                    frame,
                    f"{pgn_col}{pgn_row}",
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def draw_projected_grid(
        self, frame, rmin, rmax, cmin, cmax, color=(0, 255, 255), thick=1
    ):
        """Draws the projected grid lines on the frame."""
        if self.H is None:
            return

        def warp_rc(rc):
            xy = rc[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
            uv = cv2.perspectiveTransform(xy, self.H).reshape(-1, 2).astype(int)
            return uv

        for c in range(cmin, cmax + 1):
            p = np.array([[rmin, c], [rmax, c]], np.float32) -0.5
            u = warp_rc(p)
            cv2.line(frame, tuple(u[0]), tuple(u[1]), color, thick, cv2.LINE_AA)

        for r in range(rmin, rmax + 1):
            p = np.array([[r, cmin], [r, cmax]], np.float32) -0.5
            u = warp_rc(p)
            cv2.line(frame, tuple(u[0]), tuple(u[1]), color, thick, cv2.LINE_AA)

    def _project_rc_centers(self, rows, cols):
        """Projects all ideal grid centers to image coordinates."""
        if self.H is None:
            return None
        rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float32)
        rc = np.stack([rr, cc], axis=-1).reshape(-1, 2)
        xy = rc[:, ::-1].copy().reshape(-1, 1, 2)
        uv = cv2.perspectiveTransform(xy, self.H).reshape(-1, 2)
        return uv.reshape(rows, cols, 2)

    @staticmethod
    def _edge_len_sq(p, q):
        d = p - q
        return float(d[0] * d[0] + d[1] * d[1])

    @staticmethod
    def _longest_edge_local(p0, p1, p2):
        e01 = ChessBoard._edge_len_sq(p0, p1)
        e12 = ChessBoard._edge_len_sq(p1, p2)
        e20 = ChessBoard._edge_len_sq(p2, p0)
        lens = [(e01, (0, 1)), (e12, (1, 2)), (e20, (2, 0))]
        lens.sort(key=lambda x: (x[0], x[1]))
        return lens[-1][1]

    @staticmethod
    def _order_ccw(poly_xy):
        c = poly_xy.mean(axis=0)
        ang = np.arctan2(poly_xy[:, 1] - c[1], poly_xy[:, 0] - c[0])
        return np.argsort(ang)


def main():
    cap = cv2.VideoCapture("data/2_Move_rotate_student.mp4")
    # cap = cv2.VideoCapture("data/2_move_student.mp4")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read frame")
    
    board = ChessBoard()
    draw_frame = frame.copy()
    if board.find_board2(frame):
        print(f"Board found with orientation: {board.orientation}")
        if board.rotation is not None:
            draw_frame = cv2.rotate(draw_frame, board.rotation)
        # Draw the results
        board.draw_projected_centers(draw_frame, 8, 8, color=(0, 255, 0))
        board.draw_projected_grid(draw_frame, 0, 8, 0, 8, color=(0, 255, 255))
    else:
        print("Could not find chessboard.")
        # Draw detected corners even if board not found
        for xc, yc in board.centers:
            cv2.circle(draw_frame, (int(xc), int(yc)), 3, (0, 0, 255), -1)
    cv2.imshow("Chessboard Detection", draw_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
