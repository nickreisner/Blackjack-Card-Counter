# Blackjack Basic Strategy with Player Card Detection

This script converts the Jupyter notebook functionality into a command-line tool that can detect and identify playing cards from images using computer vision techniques, and provides blackjack strategy recommendations.

## Description

A computer vision-based blackjack card detection system that analyzes playing card images to provide optimal strategy recommendations and card counting information.

## Rules
This project assumes the following rules:
- 3:2 blackjack payout
- Double after split allowed
- Dealer hits on soft 17
- No late surrender
- Double on any two cards
- 4-8 decks

## Features

- **Edge Detection**: Uses Canny edge detection to find card boundaries
- **Card Detection**: Identifies parallelogram-shaped contours that represent cards
- **Card Extraction**: Extracts individual cards from the image
- **Template Matching**: Matches extracted cards against predefined templates
- **Blackjack Analysis**: Provides player vs dealer card analysis and strategy recommendations
- **Card Counting**: Tracks running count and true count for betting suggestions
- **Multiple Output Options**: Various plotting options to visualize different stages of the process

## Requirements

Make sure you have the following dependencies installed:
- numpy
- opencv-python (cv2)
- matplotlib
- scipy

You can install them using:
```bash
pip install numpy opencv-python matplotlib scipy
```

## Usage

### Basic Usage

#### Single Image Analysis
```bash
python blackjack_detector.py <image_path>
```

#### Interactive Mode (Recommended)
```bash
python blackjack_detector.py
# or
python blackjack_detector.py --interactive
```

Interactive mode allows you to:
- Process multiple images in a session
- Maintain running count between hands
- Set number of decks
- View current count and betting suggestions
- Process images with different flags

By default, the script will:
- Detect and identify all cards in the image
- Analyze the blackjack hand (dealer vs player cards)
- Provide strategy recommendations (Hit, Stand, Double, Split)
- Show card counting information
- Display a HUD with the analysis

### Command Line Options

- `--interactive`, `-i`: Start interactive mode
- `--is-end`: Indicate if this is the end of a hand (all cards visible)
- `--running-count`: Current running count (default: 0)
- `--num-decks`: Number of decks in play (default: 1)
- `--show-original`: Show the original input image
- `--show-edges`: Show the detected edges
- `--show-contours`: Show all detected contours
- `--show-corners`: Show detected parallelogram corners
- `--show-cards`: Show extracted individual cards
- `--show-matches`: Show card matching results
- `--save-plots`: Save plots to files instead of displaying
- `--output-dir`: Directory to save plots (default: 'output')
- `--min-score`: Minimum matching score threshold (default: 0.5)
- `--min-area`: Minimum contour area for card detection (default: 4000)

### Interactive Mode Commands

When in interactive mode, you can use these commands:

- `<image_path> [flags]` - Process an image with optional flags
  - `--is-end` - Indicate end of hand
  - `--show-plots` - Show HUD visualization
  - `--min-score=<value>` - Set minimum matching score
  - `--min-area=<value>` - Set minimum contour area
- `count` - Show current running count and betting suggestion
- `reset` - Reset running count to 0
- `decks <number>` - Set number of decks in play
- `help` - Show available commands
- `quit` or `exit` - Exit interactive mode
- `Ctrl+C` - Exit interactive mode

### Examples

1. **Interactive mode (recommended for multiple hands):**
```bash
python blackjack_detector.py
# Then in the interactive prompt:
blackjack> hands/deck1_hand7_start.jpg
blackjack> hands/deck1_hand7_end.jpg --is-end
blackjack> count
blackjack> decks 6
blackjack> quit
```

2. **Single image analysis:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg
```

3. **End of hand analysis (all cards visible):**
```bash
python blackjack_detector.py hands/deck1_hand7_end.jpg --is-end
```

4. **With custom running count:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg --running-count 5 --num-decks 6
```

4. **Show all visualizations interactively:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg --show-original --show-edges --show-contours --show-corners --show-cards --show-matches
```

5. **Save all plots to a directory:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg --show-original --show-edges --show-contours --show-corners --show-cards --show-matches --save-plots --output-dir my_results
```

6. **Only show card matching results:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg --show-matches
```

7. **Adjust detection parameters:**
```bash
python blackjack_detector.py hands/deck1_hand7_start.jpg --min-score 0.7 --min-area 5000
```

## Output

The script provides:
- **Console output**: Blackjack hand analysis including dealer/player cards, strategy recommendations, and card counting information
- **HUD display**: Visual overlay showing the analysis on the original image
- **Interactive plots**: If `--save-plots` is not used, plots are displayed interactively
- **Saved images**: If `--save-plots` is used, plots are saved as PNG files

### Default Console Output
```
=== BLACKJACK HAND ANALYSIS ===
Dealer hand: 7
Player hand: ['A', '8']
Strategy: Stand
Running count: 0
True count: 0.0

=== DETECTION SUMMARY ===
Total cards detected: 4
Total contours found: 114
Valid parallelogram contours: 4
Successfully matched cards: 3
```

### Generated Files (when using --save-plots)

- `hud_analysis.png`: HUD overlay with game analysis
- `original_image.png`: The input image
- `detected_edges.png`: Edge detection results
- `all_contours.png`: All detected contours
- `parallelogram_corners.png`: Detected card corners
- `card_1.png`, `card_2.png`, etc.: Individual extracted cards
- `card_matching_1.png`, `card_matching_2.png`, etc.: Card matching results showing the original region, isolated number, and matched template

## Template Files

The script expects template images in the `extracted_numbers/` directory:
- Two.png, Three.png, Four.png, Five.png, Six.png, Seven.png, Eight.png, Nine.png, Ten.png
- Jack.png, Queen.png, King.png, Ace.png

These templates are used for card identification through template matching.

## Algorithm Overview

1. **Preprocessing**: Convert image to grayscale and apply thresholding
2. **Edge Detection**: Use Canny edge detection to find card boundaries
3. **Contour Detection**: Find all contours in the edge image
4. **Card Filtering**: Filter contours to find parallelogram-shaped cards
5. **Card Extraction**: Extract individual cards using perspective transformation
6. **Template Matching**: Match extracted cards against predefined templates
7. **Blackjack Analysis**: Analyze dealer vs player hands and provide strategy recommendations
8. **Card Counting**: Track running count and calculate true count for betting suggestions
9. **Result Display**: Show HUD and console output with game analysis

## Notes

- The script works best with clear, well-lit images of playing cards
- Cards should be roughly parallelogram-shaped (allowing for perspective distortion)
- The minimum area parameter can be adjusted based on image resolution and card size
- The minimum score parameter controls the confidence threshold for card matching
- Use `--is-end` flag when analyzing images where all cards are visible (end of hand)
- The script automatically distinguishes between dealer and player cards based on position
- Strategy recommendations follow basic blackjack strategy rules
- Card counting uses the Hi-Lo system (2-6: +1, 7-9: 0, 10-A: -1) 