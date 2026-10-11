# Guitar Real-Time Overlay

Real-time AR overlay of scales/notes on a guitar fretboard from a close-up camera (home-office practice setup).

Approach: a custom-trained, fast fretboard keypoint model produces anchor points (nut / fret-wire × outer strings), which are smoothed over time and used by a renderer to draw the selected scale onto the neck in perspective.

- Plan: [docs/PLAN.md](docs/PLAN.md)
- Raw video data lives outside git (Google Drive + local `data/`, gitignored).
