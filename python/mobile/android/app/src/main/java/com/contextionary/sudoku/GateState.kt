package com.contextionary.sudoku

enum class GateState {
    NONE,   // no ROI yet
    L1,     // detector only (ROI found)
    L2,     // 100-pt lattice stable (intersection CNN)
    L3      // lattice still stable + rectified + classified → capture
}