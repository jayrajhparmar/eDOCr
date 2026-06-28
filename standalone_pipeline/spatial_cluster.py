def are_boxes_close(r1, r2, x_factor=2.0, y_factor=0.1):
    """
    Checks if two bounding boxes are close enough to be merged.
    r1, r2 are [x1, y1, x2, y2]
    x_factor controls how far apart words can be horizontally and still merge (e.g. same line).
    y_factor controls how far apart lines can be vertically and still merge (e.g. stacked notes).
    """
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    
    h1 = y2_1 - y1_1
    h2 = y2_2 - y1_2
    h = (h1 + h2) / 2.0
    
    # We pad the boxes outward by the threshold
    x_pad = h * x_factor / 2.0
    y_pad = h * y_factor / 2.0
    
    # Inflated bounds for intersection test
    ix1_1, iy1_1, ix2_1, iy2_1 = x1_1 - x_pad, y1_1 - y_pad, x2_1 + x_pad, y2_1 + y_pad
    ix1_2, iy1_2, ix2_2, iy2_2 = x1_2 - x_pad, y1_2 - y_pad, x2_2 + x_pad, y2_2 + y_pad
    
    # Do the inflated bounding boxes intersect?
    intersect = not (ix2_1 < ix1_2 or ix2_2 < ix1_1 or iy2_1 < iy1_2 or iy2_2 < iy1_1)
    return intersect


def merge_spatial_boxes(boxes_xywh, x_factor=3.0, y_factor=0.0):
    """
    Groups bounding boxes into cohesive horizontal lines of text.
    Enforces BOTH vertical overlap AND a maximum horizontal distance.
    """
    if not boxes_xywh:
        return []
    
    # 1. Convert to [x1, y1, x2, y2]
    rects = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes_xywh]
    
    # 2. Build adjacency matrix based on vertical overlap AND horizontal distance
    n = len(rects)
    adj = {i: set() for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            r1 = rects[i]
            r2 = rects[j]
            
            # Check vertical overlap
            h1 = r1[3] - r1[1]
            h2 = r2[3] - r2[1]
            min_h = min(h1, h2)
            
            overlap_top = max(r1[1], r2[1])
            overlap_bottom = min(r1[3], r2[3])
            overlap_h = overlap_bottom - overlap_top
            
            # Check horizontal gap
            if r1[2] < r2[0]:
                gap_x = r2[0] - r1[2]
            elif r2[2] < r1[0]:
                gap_x = r1[0] - r2[2]
            else:
                gap_x = 0 # they overlap horizontally
                
            # MERGE CONDITION:
            # 1. They must overlap vertically by at least 30% of their height
            # 2. The horizontal gap between them must be less than x_factor * height
            if overlap_h > min_h * 0.3 and gap_x < min_h * x_factor:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Find connected components
    visited = set()
    clusters = []
    
    for i in range(n):
        if i not in visited:
            cluster = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(cluster)
            
    # 4. Create final bounding boxes for each cluster
    final_boxes = []
    for cluster in clusters:
        min_x = min(rects[idx][0] for idx in cluster)
        min_y = min(rects[idx][1] for idx in cluster)
        max_x = max(rects[idx][2] for idx in cluster)
        max_y = max(rects[idx][3] for idx in cluster)
        final_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])
        
    return final_boxes
