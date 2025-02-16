import streamlit as st
import numpy as np
import time
import random

def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    return True

def solve_sudoku_blind(board, steps):
    empty = find_empty(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            steps.append((row, col, num))
            if solve_sudoku_blind(board, steps):
                return True
            board[row, col] = 0
    return False

def solve_sudoku_heuristic(board, steps):
    empty = find_empty(board)
    if not empty:
        return True
    row, col = empty
    nums = list(range(1, 10))
    random.shuffle(nums)
    for num in nums:
        if is_valid(board, row, col, num):
            board[row, col] = num
            steps.append((row, col, num))
            if solve_sudoku_heuristic(board, steps):
                return True
            board[row, col] = 0
    return False

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return (i, j)
    return None

def generate_sudoku():
    board = np.zeros((9, 9), dtype=int)
    filled_positions = random.sample(range(81), 17)
    for pos in filled_positions:
        row, col = divmod(pos, 9)
        num = random.randint(1, 9)
        if is_valid(board, row, col, num):
            board[row, col] = num
    return board

def draw_board(board):
    table_html = "<table style='border-collapse: collapse; font-size: 20px;'>"
    for i in range(9):
        table_html += "<tr>"
        for j in range(9):
            num = board[i, j]
            style = "border: 1px solid black; width: 30px; height: 30px; text-align: center;"
            if (i % 3 == 0 and i != 0):
                style += "border-top: 3px solid black;"
            if (j % 3 == 0 and j != 0):
                style += "border-left: 3px solid black;"
            table_html += f"<td style='{style}'>{num if num != 0 else ''}</td>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html

st.title("Sudoku Solver with Blind & Heuristic Search")

if "original_board" not in st.session_state:
    st.session_state.original_board = generate_sudoku()
    st.session_state.working_board = np.copy(st.session_state.original_board)
    st.session_state.solved_blind = None
    st.session_state.solved_heuristic = None
    st.session_state.steps_blind = []
    st.session_state.steps_heuristic = []

t1, t2 = st.columns(2)
with t1:
    st.markdown("### Initial Sudoku")
    st.markdown(draw_board(st.session_state.original_board), unsafe_allow_html=True)

if st.button("Solve with Blind Search"):
    board_blind = np.copy(st.session_state.working_board)
    st.session_state.steps_blind = []
    start_time = time.time()
    solve_sudoku_blind(board_blind, st.session_state.steps_blind)
    st.session_state.solved_blind = board_blind
    st.session_state.time_blind = time.time() - start_time

if st.button("Solve with Heuristic Search"):
    board_heuristic = np.copy(st.session_state.working_board)
    st.session_state.steps_heuristic = []
    start_time = time.time()
    solve_sudoku_heuristic(board_heuristic, st.session_state.steps_heuristic)
    st.session_state.solved_heuristic = board_heuristic
    st.session_state.time_heuristic = time.time() - start_time

if st.button("Reset Board"):
    st.session_state.original_board = generate_sudoku()
    st.session_state.working_board = np.copy(st.session_state.original_board)
    st.session_state.solved_blind = None
    st.session_state.solved_heuristic = None
    st.session_state.steps_blind = []
    st.session_state.steps_heuristic = []

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Blind Search Solution")
    if st.session_state.solved_blind is not None:
        st.markdown(draw_board(st.session_state.solved_blind), unsafe_allow_html=True)
        st.write(f"Runtime: {st.session_state.time_blind:.4f} sec")
        st.write("Steps:", st.session_state.steps_blind)

with c2:
    st.markdown("### Heuristic Search Solution")
    if st.session_state.solved_heuristic is not None:
        st.markdown(draw_board(st.session_state.solved_heuristic), unsafe_allow_html=True)
        st.write(f"Runtime: {st.session_state.time_heuristic:.4f} sec")
        st.write("Steps:", st.session_state.steps_heuristic)
