# sudoku_tool_api.py
# Optional FastAPI wrapper for the tool functions.
# Run with: uvicorn sudoku_tool_api:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from sudoku_tools import sanity_check, compute_candidates_tool, next_moves as _next_moves, apply_move as _apply_move

app = FastAPI(title="Sudoku Companion Tool API")

class GridModel(BaseModel):
    grid: List[List[int]]

class CandidatesModel(BaseModel):
    candidates: Dict[str, List[int]]

class NextMovesRequest(BaseModel):
    current: List[List[int]]
    candidates: Optional[Dict[str, List[int]]] = None
    max_difficulty: str = "hidden_single"
    max_moves: int = 3

class MoveModel(BaseModel):
    cell: str
    digit: int

class ApplyMoveRequest(BaseModel):
    current: List[List[int]]
    move: MoveModel

@app.post("/sanity_check")
def api_sanity(payload: Dict[str, List[List[int]]]):
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
