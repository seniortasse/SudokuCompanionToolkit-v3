# sudoku_tool_api.py
# Optional FastAPI wrapper for the tool functions.
# Run with: uvicorn sudoku_tool_api:app --reload

from fastapi import FastAPI
from pydantic import BaseModel

from sudoku_tools import apply_move as _apply_move
from sudoku_tools import compute_candidates_tool, sanity_check
from sudoku_tools import next_moves as _next_moves

app = FastAPI(title="Sudoku Companion Tool API")


class GridModel(BaseModel):
    grid: list[list[int]]


class CandidatesModel(BaseModel):
    candidates: dict[str, list[int]]


class NextMovesRequest(BaseModel):
    current: list[list[int]]
    candidates: dict[str, list[int]] | None = None
    max_difficulty: str = "hidden_single"
    max_moves: int = 3


class MoveModel(BaseModel):
    cell: str
    digit: int


class ApplyMoveRequest(BaseModel):
    current: list[list[int]]
    move: MoveModel


@app.post("/sanity_check")
def api_sanity(payload: dict[str, list[list[int]]]):
    return sanity_check(payload["original"], payload["current"])


@app.post("/compute_candidates")
def api_cands(payload: GridModel):
    return compute_candidates_tool(payload.grid)


@app.post("/next_moves")
def api_moves(req: NextMovesRequest):
    return _next_moves(req.current, req.candidates, req.max_difficulty, req.max_moves)


@app.post("/apply_move")
def api_apply(req: ApplyMoveRequest):
    return _apply_move(req.current, req.move.dict())
