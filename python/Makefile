PY?=python
IMG?samples/sample_puzzle.jpg
OUT?=demo_export

.PHONY: install demo storyboard gif mp4 all clean

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) apps/cli/demo_cli_overlay.py --image $(IMG) --out $(OUT) --mode demo --max_moves 5 > $(OUT)/moves.json

storyboard:
	$(PY) apps/cli/storyboard_sheet.py --dir $(OUT) --out $(OUT)/storyboard --paper letter --cols 2 --json $(OUT)/moves.json

gif:
	$(PY) apps/cli/animate_gif.py --dir $(OUT) --out $(OUT)/moves.gif

mp4:
	$(PY) apps/cli/animate_mp4.py --dir $(OUT) --out $(OUT)/moves.mp4

all: install demo storyboard gif mp4

clean:
	rm -rf $(OUT)