#!/usr/bin/env python3
"""
Blackjack Card Detection Script

This script detects and identifies playing cards from images using computer vision techniques.
It can be run from the command line with various options to control output plots.

Usage:
    python blackjack_detector.py <image_path> [options]

Arguments:
    image_path    Path to the image file to process

Options:
    --show-original     Show the original input image
    --show-edges        Show the detected edges
    --show-contours     Show all detected contours
    --show-corners      Show detected parallelogram corners
    --show-cards        Show extracted individual cards
    --show-matches      Show card matching results
    --save-plots        Save plots to files instead of displaying
    --output-dir        Directory to save plots (default: 'output')
    --min-score         Minimum matching score threshold (default: 0.5)
    --min-area          Minimum contour area for card detection (default: 4000)
    --help              Show this help message
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path for basic_strategy import
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
from basic_strategy import basic_strategy


def preprocess_image(image, display=False):
    """Preprocess image for edge detection."""
    if image is None:
        raise FileNotFoundError(f"Error: no image found")

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_threshold = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    if display:
        plt.imshow(img_threshold, cmap='gray')
        plt.title('Processed input image')
        plt.axis('off')
        plt.show()

    return img_threshold


def gaussian_kernel(size, sigma):
    """Create a Gaussian kernel for smoothing."""
    kernel = np.zeros((size, size))
    k = (size - 1) / 2
    sigma_squared = np.square(sigma)
    norm_constant = 1 / (2 * np.pi * sigma_squared)

    for i in range(size):
        for j in range(size):
            exponent = np.exp(-(np.square(i-k) + np.square(j-k)) / (2 * sigma_squared))
            kernel[i, j] = norm_constant * exponent

    return kernel


def conv(image, kernel):
    """Perform 2D convolution."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width = ((Hk // 2, Hk // 2), (Wk // 2, Wk // 2))
    padded = np.pad(image, pad_width, mode='edge')
    flipped_kernel = np.flip(kernel)

    for m in range(Hi):
        for n in range(Wi):
            kernel_region = padded[m:m+Hk, n:n+Wk]
            out[m, n] = np.sum(kernel_region * flipped_kernel)

    return out


def partial_x(img):
    """Compute partial derivative in x direction."""
    D_x = np.array([[0.5, 0, -0.5]])
    return conv(img, D_x)


def partial_y(img):
    """Compute partial derivative in y direction."""
    D_y = np.array([[0.5], [0], [-0.5]])
    return conv(img, D_y)


def gradient(img):
    """Compute gradient magnitude and direction."""
    G_partial_x = partial_x(img)
    G_partial_y = partial_y(img)

    G = np.sqrt(np.square(G_partial_x) + np.square(G_partial_y))
    theta = np.degrees(np.arctan2(G_partial_y, G_partial_x)) % 360

    return G, theta


def non_maximum_suppression(G, theta, soft_threshold=0.7):
    """Perform non-maximum suppression on gradient image."""
    H, W = G.shape
    out = np.zeros((H, W))

    theta = np.floor((theta + 22.5) / 45) * 45

    neighbor_shifts = {
        0: [(0, 1), (0, -1)],
        45: [(-1, -1), (1, 1)],
        90: [(1, 0), (-1, 0)],
        135: [(-1, 1), (1, -1)]
    }

    for i in range(H):
        for j in range(W):
            current = G[i, j]
            direction = theta[i, j] % 180
            shift1, shift2 = neighbor_shifts[direction]

            neighbor1_idx = (i + shift1[0], j + shift1[1])
            neighbor2_idx = (i + shift2[0], j + shift2[1])

            neighbor1 = G[neighbor1_idx] if 0 <= neighbor1_idx[0] < H and 0 <= neighbor1_idx[1] < W else 0
            neighbor2 = G[neighbor2_idx] if 0 <= neighbor2_idx[0] < H and 0 <= neighbor2_idx[1] < W else 0

            if current >= neighbor1 * soft_threshold and current >= neighbor2 * soft_threshold:
                out[i, j] = current

    return out


def double_thresholding(img, high, low):
    """Apply double thresholding to create strong and weak edges."""
    strong_edges = img > high
    weak_edges = (img <= high) & (img > low)
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """Get 8-connected neighbors of a pixel."""
    neighbors = []
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if 0 <= i < H and 0 <= j < W and (i != y or j != x):
                neighbors.append((i, j))
    return neighbors


def link_edges(strong_edges, weak_edges):
    """Link strong and weak edges using hysteresis."""
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.copy(strong_edges)
    weak_edges = np.copy(weak_edges)

    for i, j in indices:
        neighbor_list = [(i, j)]

        while neighbor_list:
            y, x = neighbor_list.pop(0)
            for neighbor in get_neighbors(y, x, H, W):
                if weak_edges[neighbor]:
                    edges[neighbor] = True
                    weak_edges[neighbor] = False
                    neighbor_list.append(neighbor)

    return edges


def max_pooling(img, kernel_size=3):
    """Apply max pooling to close gaps after linking edges."""
    H, W = img.shape
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    out = np.zeros((H,W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            out[i, j] = np.max(padded[i:i + kernel_size, j:j+kernel_size])
    return out


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """Complete Canny edge detection pipeline."""
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    edge = max_pooling(edge, kernel_size=3)
    return edge


def order_points(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_card(image, pts):
    """Extract a card from an image using corner points."""
    target_width = 300
    target_height = int(target_width / 0.714)  # fixed aspect ratio
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped


def get_approximated_parallelogram(contour, epsilon_ratio=0.02, angle_threshold=15,
                                    min_aspect_ratio=0.2, max_aspect_ratio=10, min_area=4000):
    """Approximate contour as a parallelogram and return ordered corner points."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon_ratio * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2)
    ordered_pts = order_points(pts)
    
    area = cv2.contourArea(ordered_pts.reshape((-1,1,2)).astype(np.int32))
    if area < min_area:
        return None

    vecs = []
    for i in range(4):
        pt1 = ordered_pts[i]
        pt2 = ordered_pts[(i + 1) % 4]
        vecs.append(pt2 - pt1)
    vecs = np.array(vecs)
    
    def angle_between(v1, v2):
        dot = np.dot(v1, v2)
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod == 0:
            return 0
        cos_angle = dot / norm_prod
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    angle1 = angle_between(vecs[0], vecs[2])
    angle2 = angle_between(vecs[1], vecs[3])
    
    def is_parallel(angle, threshold):
        return min(angle, abs(180 - angle)) < threshold
    
    if not (is_parallel(angle1, angle_threshold) and is_parallel(angle2, angle_threshold)):
        return None
    
    rect = cv2.minAreaRect(approx)
    (center, (width, height), _) = rect
    if width == 0 or height == 0:
        return None
    aspect_ratio = min(width, height) / max(width, height)
    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return None
    
    return ordered_pts


def extract_top_left_region(card, fraction=0.2):
    """Extract top left corner region of a card."""
    h, w = card.shape[:2]
    crop_h = int(h * fraction)
    crop_w = int(w * fraction)
    return card[:crop_h, :crop_w]


def normalized_threshold_region(region, thresh_value=128):
    """Normalize and threshold a region."""
    if len(region.shape) == 3:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        region_gray = region
    norm_img = cv2.normalize(region_gray, None, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(norm_img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def isolate_number_region(region, min_area=50, crop_ratio_threshold=0.9, debug=False):
    """Isolate the number from a binary region."""
    inv = 255 - region
    contours, _ = cv2.findContours(inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = region.shape
    region_area = h * w
    valid_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if x <= 0 or y <= 0 or (x+cw) >= w or (y+ch) >= h:
            continue
        if cv2.contourArea(cnt) < min_area:
            continue
        valid_contours.append(cnt)
    if len(valid_contours) == 0:
        if debug:
            print("No valid contour found.")
        return None
    
    best_cnt = max(valid_contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(best_cnt)
    cropped = region[y:y+ch, x:x+cw]
    
    crop_area = cw * ch
    if crop_area > crop_ratio_threshold * region_area:
        if debug:
            print("Crop did not reduce region significantly; likely back of card.")
        return None
    
    if debug:
        plt.figure(figsize=(4,3))
        plt.imshow(cropped, cmap="gray")
        plt.title("Isolated Number")
        plt.axis("off")
        plt.show()
    
    return cropped


def match_card_template(card, templates, min_score=0.3):
    """Match a card against templates using template matching."""
    raw_top_left = extract_top_left_region(card, fraction=0.2)
    thresh_region = normalized_threshold_region(raw_top_left, thresh_value=180)
    number_region = isolate_number_region(thresh_region, min_area=50, crop_ratio_threshold=0.9, debug=False)
    
    if number_region is None:
        return None, None, None

    best_label = None
    best_score = -1

    for label, tmpl in templates.items():
        resized_tmpl = cv2.resize(tmpl, (number_region.shape[1], number_region.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = cv2.matchTemplate(number_region, resized_tmpl, cv2.TM_CCOEFF_NORMED)
        score = result[0][0]
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score, number_region


def load_templates():
    """Load card templates from the extracted_numbers directory."""
    template_files = [
        "extracted_numbers/Two.png", "extracted_numbers/Three.png", "extracted_numbers/Four.png", 
        "extracted_numbers/Five.png", "extracted_numbers/Six.png", "extracted_numbers/Seven.png", 
        "extracted_numbers/Eight.png", "extracted_numbers/Nine.png", "extracted_numbers/Ten.png",
        "extracted_numbers/Jack.png", "extracted_numbers/Queen.png", "extracted_numbers/King.png", 
        "extracted_numbers/Ace.png"
    ]

    templates = {}
    for file in template_files:
        tmpl = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            print(f"Warning: Could not load template image: {file}")
            continue
        label = file.split('/')[-1].split('.')[0]
        templates[label] = tmpl
    
    return templates


def save_or_show_plot(fig, filename, save_plots, output_dir):
    """Save or show a plot based on the save_plots flag."""
    if save_plots:
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def count_value(card_label):
    """Get the card counting value for a card."""
    label_map = {
        'Two': 1, 'Three': 1, 'Four': 1, 'Five': 1, 'Six': 1,
        'Seven': 0, 'Eight': 0, 'Nine': 0,
        'Ten': -1, 'Jack': -1, 'Queen': -1, 'King': -1, 'Ace': -1
    }
    return label_map.get(card_label, 0)


def get_bet_suggestion(true_count):
    """Get betting suggestion based on true count."""
    if true_count < 0:
        return "Min bet"
    elif true_count < 2:
        return "1x bet"
    elif true_count < 4:
        return "2x bet"
    else:
        return "3x+ bet"


def display_HUD(image, is_end, dealer_hand, player_cards, strategy, running_cnt, true_cnt, save_plots=False, output_dir='output', parallelogram_corners=None):
    """Display HUD with game information."""
    # HUD display space
    h, w, _ = image.shape
    HUD_space = 450
    full_width = w + HUD_space
    HUD_img = np.ones((h, full_width, 3), dtype=np.uint8) * 255
    
    # Create image with card outlines if corners are provided
    if parallelogram_corners is not None:
        img_with_outlines = image.copy()
        for pts in parallelogram_corners:
            pts_int = pts.astype(int)
            # Draw the card outline
            cv2.line(img_with_outlines, tuple(pts_int[0]), tuple(pts_int[1]), (0, 255, 0), 2)
            cv2.line(img_with_outlines, tuple(pts_int[1]), tuple(pts_int[2]), (0, 255, 0), 2)
            cv2.line(img_with_outlines, tuple(pts_int[2]), tuple(pts_int[3]), (0, 255, 0), 2)
            cv2.line(img_with_outlines, tuple(pts_int[3]), tuple(pts_int[0]), (0, 255, 0), 2)
        HUD_img[:, :w] = img_with_outlines
    else:
        HUD_img[:, :w] = image
    
    x = w + 20
    
    # image parameters
    y_offset = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2

    # display dealer card info
    if is_end:
        dealer_hand_str = ', '.join(dealer_hand)
    else:
        dealer_hand_str = dealer_hand
    cv2.putText(HUD_img, f"Dealer hand: {dealer_hand_str}", (x, y_offset), font, font_scale, color, thickness)
    y_offset += 100
    
    # display player card info
    player_cards_str = ', '.join(player_cards)
    cv2.putText(HUD_img, f"Player hand: {player_cards_str}", (x, y_offset), font, font_scale, color, thickness)
    y_offset += 100
    
    # display strategy
    if not is_end:
        cv2.putText(HUD_img, f"Strategy: {strategy}", (x, y_offset), font, font_scale, color, thickness)
        y_offset += 100
    
    cv2.putText(HUD_img, f"Running Count: {running_cnt}", (x, y_offset), font, font_scale, color, thickness)
    y_offset += 50
    cv2.putText(HUD_img, f"True Count: {true_cnt:.1f}", (x, y_offset), font, font_scale, color, thickness)
    y_offset += 50

    if is_end:
        bet_suggestion = get_bet_suggestion(true_cnt)
        cv2.putText(HUD_img, f"Bet Suggestion: {bet_suggestion}", (x, y_offset), font, font_scale, color, thickness)
        y_offset += 50
    
    # show HUD
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(cv2.cvtColor(HUD_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Blackjack Hand Analysis")
    ax.axis("off")
    plt.tight_layout()
    
    if save_plots:
        output_path = os.path.join(output_dir, "hud_analysis.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved HUD analysis to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def interactive_mode():
    """Interactive mode for processing multiple images with persistent state."""
    print("=== BLACKJACK DETECTOR INTERACTIVE MODE ===")
    print("Commands:")
    print("  <image_path> [flags]  - Process an image")
    print("  count                 - Show current running count")
    print("  reset                 - Reset running count to 0")
    print("  decks <number>        - Set number of decks")
    print("  help                  - Show this help")
    print("  quit/exit             - Exit interactive mode")
    print("  Ctrl+C                - Exit interactive mode")
    print()
    
    # Initialize state
    running_count = 0
    num_decks = 1
    templates = load_templates()
    
    if not templates:
        print("Error: No templates loaded. Make sure the extracted_numbers directory exists with template images.")
        return
    
    print(f"Initial state: Running count = {running_count}, Decks = {num_decks}")
    print()
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("blackjack> ").strip()
                
                if not user_input:
                    continue
                
                # Parse the input
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                elif command == 'help':
                    print("Commands:")
                    print("  <image_path> [flags]  - Process an image")
                    print("  count                 - Show current running count")
                    print("  reset                 - Reset running count to 0")
                    print("  decks <number>        - Set number of decks")
                    print("  help                  - Show this help")
                    print("  quit/exit             - Exit interactive mode")
                elif command == 'count':
                    true_count = running_count // num_decks
                    print(f"Running count: {running_count}")
                    print(f"True count: {true_count:.1f}")
                    bet_suggestion = get_bet_suggestion(true_count)
                    print(f"Bet suggestion: {bet_suggestion}")
                elif command == 'reset':
                    running_count = 0
                    print(f"Running count reset to {running_count}")
                elif command == 'decks':
                    if len(parts) < 2:
                        print("Usage: decks <number>")
                        continue
                    try:
                        num_decks = int(parts[1])
                        print(f"Number of decks set to {num_decks}")
                    except ValueError:
                        print("Invalid number of decks")
                else:
                    # Assume it's an image path with optional flags
                    image_path = parts[0]
                    
                    # Parse flags
                    is_end = False
                    show_plots = False
                    min_score = 0.5
                    min_area = 4000
                    
                    for part in parts[1:]:
                        if part == '--is-end':
                            is_end = True
                        elif part == '--show-plots':
                            show_plots = True
                        elif part.startswith('--min-score='):
                            try:
                                min_score = float(part.split('=')[1])
                            except ValueError:
                                print("Invalid min-score value")
                        elif part.startswith('--min-area='):
                            try:
                                min_area = int(part.split('=')[1])
                            except ValueError:
                                print("Invalid min-area value")
                    
                    # Process the image
                    print(f"Processing: {image_path}")
                    print(f"Flags: is_end={is_end}, show_plots={show_plots}")
                    
                    # Load image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error: Could not load image from {image_path}")
                        continue
                    
                    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Detect edges
                    edges = canny(img_gray)
                    
                    # Find contours and detect cards
                    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    parallelogram_corners = []
                    for contour in contours:
                        pts = get_approximated_parallelogram(contour, min_area=min_area)
                        if pts is not None:
                            parallelogram_corners.append(pts)
                    
                    # Extract cards
                    cards = []
                    for pts in parallelogram_corners:
                        card = extract_card(img_gray, pts)
                        if card is not None:
                            cards.append(card)
                    
                    # Match cards and get their positions
                    matches = []
                    for idx, card in enumerate(cards):
                        best_label, best_score, number_region = match_card_template(card, templates, min_score=min_score)
                        
                        if best_label is not None:
                            if len(parallelogram_corners) > idx:
                                tl = parallelogram_corners[idx][0]
                                matches.append((best_label, (int(tl[0]), int(tl[1]))))
                    
                    # Analyze blackjack hand
                    if matches:
                        analysis = analyze_blackjack_hand(matches, is_end, running_count, num_decks)
                        
                        # Update running count
                        running_count = analysis['running_count']
                        
                        # Print console output
                        print(f"\n=== BLACKJACK HAND ANALYSIS ===")
                        print(f"Dealer hand: {analysis['dealer_hand']}")
                        print(f"Player hand: {analysis['player_hand']}")
                        if not is_end:
                            print(f"Strategy: {analysis['strategy']}")
                        print(f"Running count: {analysis['running_count']}")
                        print(f"True count: {analysis['true_count']:.1f}")
                        if is_end:
                            bet_suggestion = get_bet_suggestion(analysis['true_count'])
                            print(f"Bet suggestion: {bet_suggestion}")
                        
                        # Show HUD by default in interactive mode (unless --save-plots is used)
                        display_HUD(img, analysis['is_end'], analysis['dealer_hand'], 
                                   analysis['player_hand'], analysis['strategy'], 
                                   analysis['running_count'], analysis['true_count'],
                                   parallelogram_corners=parallelogram_corners)
                        
                        print(f"Cards detected: {len(cards)}, Matched: {len(matches)}")
                    else:
                        print("No cards detected or matched!")
                    
                    print()
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nGoodbye!")


def analyze_blackjack_hand(matches, is_end, running_count=0, num_decks=1):
    """Analyze the blackjack hand and return game information."""
    if not matches:
        return {
            'dealer_hand': [],
            'player_hand': [],
            'strategy': 'No cards detected',
            'running_count': running_count,
            'true_count': running_count // num_decks,
            'is_end': is_end
        }
    
    # Extract y-coordinates for clustering
    y_coords = [match[1][1] for match in matches]
    
    # Simple clustering: group cards by vertical position
    # Sort by y-coordinate and find natural breaks
    sorted_by_y = sorted(matches, key=lambda x: x[1][1])
    y_coords_sorted = [match[1][1] for match in sorted_by_y]
    
    # Find the largest gap in y-coordinates to separate dealer and player cards
    if len(y_coords_sorted) > 1:
        gaps = []
        for i in range(1, len(y_coords_sorted)):
            gaps.append(y_coords_sorted[i] - y_coords_sorted[i-1])
        
        # Find the largest gap
        max_gap_idx = gaps.index(max(gaps))
        split_point = max_gap_idx + 1
        
        # Split into two groups
        upper_group = sorted_by_y[:split_point]  # Dealer cards (higher in image)
        lower_group = sorted_by_y[split_point:]  # Player cards (lower in image)
        
        dealer_hand = [match[0] for match in upper_group]
        player_hand = [match[0] for match in lower_group]
        
        # If we have very few cards, use a simpler approach
        if len(matches) <= 3:
            # For small hands, assume dealer gets the top card(s), player gets the bottom
            if len(matches) == 2:
                dealer_hand = [sorted_by_y[0][0]]  # Top card
                player_hand = [sorted_by_y[1][0]]  # Bottom card
            elif len(matches) == 3:
                dealer_hand = [sorted_by_y[0][0]]  # Top card
                player_hand = [sorted_by_y[1][0], sorted_by_y[2][0]]  # Bottom two cards
    else:
        # Only one card detected
        dealer_hand = [sorted_by_y[0][0]]
        player_hand = []
    
    # Update running count
    if is_end:
        for card in dealer_hand:
            running_count += count_value(card)
        for card in player_hand:
            running_count += count_value(card)
    
    true_count = running_count // num_decks
    
    # Get optimal strategy
    if not is_end:
        strategy = basic_strategy(player_hand, dealer_hand[0])
    else:
        strategy = "Stand"
    
    # Map cards from words to numbers/letters
    card_map = {
        'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5', 'Six': '6',
        'Seven': '7', 'Eight': '8', 'Nine': '9', 'Ten': '10',
        'Jack': 'J', 'Queen': 'Q', 'King': 'K', 'Ace': 'A'
    }
    
    if is_end:
        dealer_val = [card_map.get(card, card) for card in dealer_hand]
    else:
        dealer_val = card_map.get(dealer_hand[0], dealer_hand[0])
    player_vals = [card_map.get(card, card) for card in player_hand]
    
    return {
        'dealer_hand': dealer_val,
        'player_hand': player_vals,
        'strategy': strategy,
        'running_count': running_count,
        'true_count': true_count,
        'is_end': is_end
    }


def main():
    parser = argparse.ArgumentParser(description='Blackjack Card Detection Script')
    parser.add_argument('image_path', nargs='?', help='Path to the image file to process')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--is-end', action='store_true', help='Indicate if this is the end of a hand (all cards visible)')
    parser.add_argument('--show-original', action='store_true', help='Show the original input image')
    parser.add_argument('--show-edges', action='store_true', help='Show the detected edges')
    parser.add_argument('--show-contours', action='store_true', help='Show all detected contours')
    parser.add_argument('--show-corners', action='store_true', help='Show detected parallelogram corners')
    parser.add_argument('--show-cards', action='store_true', help='Show extracted individual cards')
    parser.add_argument('--show-matches', action='store_true', help='Show card matching results')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files instead of displaying')
    parser.add_argument('--output-dir', default='output', help='Directory to save plots (default: output)')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum matching score threshold (default: 0.5)')
    parser.add_argument('--min-area', type=int, default=4000, help='Minimum contour area for card detection (default: 4000)')
    parser.add_argument('--running-count', type=int, default=0, help='Current running count (default: 0)')
    parser.add_argument('--num-decks', type=int, default=1, help='Number of decks in play (default: 1)')
    
    args = parser.parse_args()
    
    # Check if interactive mode is requested or no image path provided
    if args.interactive or args.image_path is None:
        interactive_mode()
        return
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        sys.exit(1)
    
    img_gray = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    
    # Show original image
    if args.show_original:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("Original Input Image")
        ax.axis("off")
        save_or_show_plot(fig, "original_image.png", args.save_plots, args.output_dir)
    
    # Detect edges
    edges = canny(img_gray)
    
    # Show edges
    if args.show_edges:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(edges, cmap='gray')
        ax.set_title("Detected Edges")
        ax.axis("off")
        save_or_show_plot(fig, "detected_edges.png", args.save_plots, args.output_dir)
    
    # Find contours and detect cards
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parallelogram_corners = []
    for contour in contours:
        pts = get_approximated_parallelogram(contour, min_area=args.min_area)
        if pts is not None:
            parallelogram_corners.append(pts)
    
    # Show all contours
    if args.show_contours:
        img_all = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_all, contours, -1, (0, 255, 0), 2)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
        ax.set_title("All Detected Contours")
        ax.axis("off")
        save_or_show_plot(fig, "all_contours.png", args.save_plots, args.output_dir)
    
    # Show parallelogram corners
    if args.show_corners:
        img_with_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for pts in parallelogram_corners:
            pts_int = pts.astype(int)
            for pt in pts_int:
                cv2.circle(img_with_corners, tuple(pt), 5, (0, 0, 255), -1)
            cv2.line(img_with_corners, tuple(pts_int[0]), tuple(pts_int[1]), (0, 0, 255), 2)
            cv2.line(img_with_corners, tuple(pts_int[1]), tuple(pts_int[2]), (0, 0, 255), 2)
            cv2.line(img_with_corners, tuple(pts_int[2]), tuple(pts_int[3]), (0, 0, 255), 2)
            cv2.line(img_with_corners, tuple(pts_int[3]), tuple(pts_int[0]), (0, 0, 255), 2)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        ax.set_title("Detected Parallelogram Corners")
        ax.axis("off")
        save_or_show_plot(fig, "parallelogram_corners.png", args.save_plots, args.output_dir)
    
    # Extract cards
    cards = []
    for pts in parallelogram_corners:
        card = extract_card(img_gray, pts)
        if card is not None:
            cards.append(card)
    
    # Show extracted cards
    if args.show_cards:
        for idx, card in enumerate(cards):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(card, cmap='gray')
            ax.set_title(f"Extracted Card {idx + 1}")
            ax.axis("off")
            save_or_show_plot(fig, f"card_{idx+1}.png", args.save_plots, args.output_dir)
    
    # Load templates and match cards
    templates = load_templates()
    if not templates:
        print("Error: No templates loaded. Make sure the extracted_numbers directory exists with template images.")
        sys.exit(1)
    
    # Match cards and get their positions
    matches = []
    for idx, card in enumerate(cards):
        best_label, best_score, number_region = match_card_template(card, templates, min_score=args.min_score)
        
        if best_label is not None:
            # Get the top-left corner position of the card
            if len(parallelogram_corners) > idx:
                tl = parallelogram_corners[idx][0]  # Top-left corner
                matches.append((best_label, (int(tl[0]), int(tl[1]))))
    
    # Show matching results if requested
    if args.show_matches:
        for idx, card in enumerate(cards):
            best_label, best_score, number_region = match_card_template(card, templates, min_score=args.min_score)
            
            if number_region is None:
                print(f"Card {idx+1}: No valid number region detected (likely back of card).")
                continue
            
            print(f"Card {idx+1}: Best Template Match: {best_label} with score {best_score:.2f}")
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(extract_top_left_region(card), cmap="gray")
            axes[0].set_title("Raw Top Left")
            axes[0].axis("off")
            
            axes[1].imshow(number_region, cmap="gray")
            axes[1].set_title("Isolated Number")
            axes[1].axis("off")
            
            if best_label is not None:
                axes[2].imshow(templates[best_label], cmap="gray")
                axes[2].set_title(f"Matched Template: {best_label}")
            else:
                axes[2].text(0.5, 0.5, "No match", ha="center", va="center", fontsize=16)
            axes[2].axis("off")
            
            plt.tight_layout()
            save_or_show_plot(fig, f"card_matching_{idx+1}.png", args.save_plots, args.output_dir)
    
    # Analyze blackjack hand (default behavior)
    if matches:
        analysis = analyze_blackjack_hand(matches, args.is_end, args.running_count, args.num_decks)
        
        # Print console output
        print(f"\n=== BLACKJACK HAND ANALYSIS ===")
        print(f"Dealer hand: {analysis['dealer_hand']}")
        print(f"Player hand: {analysis['player_hand']}")
        if not args.is_end:
            print(f"Strategy: {analysis['strategy']}")
        print(f"Running count: {analysis['running_count']}")
        print(f"True count: {analysis['true_count']:.1f}")
        if args.is_end:
            bet_suggestion = get_bet_suggestion(analysis['true_count'])
            print(f"Bet suggestion: {bet_suggestion}")
        
        # Show HUD (default behavior unless --save-plots is used)
        if not args.save_plots or any([args.show_original, args.show_edges, args.show_contours, 
                                     args.show_corners, args.show_cards, args.show_matches]):
            display_HUD(img, analysis['is_end'], analysis['dealer_hand'], 
                       analysis['player_hand'], analysis['strategy'], 
                       analysis['running_count'], analysis['true_count'],
                       parallelogram_corners=parallelogram_corners)
        elif args.save_plots:
            display_HUD(img, analysis['is_end'], analysis['dealer_hand'], 
                       analysis['player_hand'], analysis['strategy'], 
                       analysis['running_count'], analysis['true_count'], 
                       save_plots=True, output_dir=args.output_dir,
                       parallelogram_corners=parallelogram_corners)
    else:
        print("No cards detected or matched!")
    
    # Print summary
    print(f"\n=== DETECTION SUMMARY ===")
    print(f"Total cards detected: {len(cards)}")
    print(f"Total contours found: {len(contours)}")
    print(f"Valid parallelogram contours: {len(parallelogram_corners)}")
    print(f"Successfully matched cards: {len(matches)}")


if __name__ == "__main__":
    main()
