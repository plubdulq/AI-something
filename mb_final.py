import streamlit as st
import numpy as np
import time
import random
import psutil
import os
import plotly.graph_objects as go

def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row+3, start_col:start_col+3]:
        return False
    return True

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return (i, j)
    return None

def find_empty_mrv(board):
    """ เลือกตำแหน่งที่มีตัวเลือกตัวเลขน้อยที่สุด (MRV) """
    min_options = float('inf')
    best_pos = None

    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:  # ถ้าเป็นช่องว่าง
                options = [num for num in range(1, 10) if is_valid(board, i, j, num)]
                if len(options) < min_options:
                    min_options = len(options)
                    best_pos = (i, j)
    
    return best_pos

def get_least_constraining_values(board, row, col):
    """ คืนลิสต์ของตัวเลขที่กระทบช่องอื่นน้อยที่สุด (LCV) """
    num_constraints = {}
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            # นับว่าหากใส่ num จะไปบล็อกช่องอื่นๆ กี่ช่อง
            count = sum(
                1 for i in range(9) for j in range(9)
                if board[i, j] == 0 and is_valid(board, i, j, num)
            )
            num_constraints[num] = count
    
    # เรียงค่าตามผลกระทบจากน้อยไปมาก (LCV)
    return sorted(num_constraints, key=num_constraints.get)

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Return memory in MB

if "memory_blind" not in st.session_state:
    st.session_state.memory_blind = []
if "memory_heuristic" not in st.session_state:
    st.session_state.memory_heuristic = []
if "memory_history_blind" not in st.session_state:
    st.session_state.memory_history_blind = []
if "memory_history_heuristic" not in st.session_state:
    st.session_state.memory_history_heuristic = []
if "total_memory_history" not in st.session_state:
    st.session_state.total_memory_history = []  # Store (blind, heuristic) tuples

def solve_sudoku_blind(board, steps):
    empty = find_empty(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            steps.append((row, col, num))
            st.session_state.memory_blind.append(memory_usage())
            if solve_sudoku_blind(board, steps):
                return True
            board[row, col] = 0
            steps.append((row, col, 0))  # Add backtracking step
            st.session_state.memory_blind.append(memory_usage())
    return False

def solve_sudoku_heuristic(board, steps):
    empty = find_empty_mrv(board)  # ใช้ MRV หา position ที่ดีที่สุด
    if not empty:
        return True

    row, col = empty
    nums = get_least_constraining_values(board, row, col)  # ใช้ LCV เลือกค่าที่ดีที่สุด

    for num in nums:
        if is_valid(board, row, col, num):
            board[row, col] = num
            steps.append((row, col, num))
            st.session_state.memory_heuristic.append(memory_usage())
            if solve_sudoku_heuristic(board, steps):
                return True
            
            board[row, col] = 0  # Backtracking
            steps.append((row, col, 0))  # Add backtracking step
            st.session_state.memory_heuristic.append(memory_usage())

    return False

def generate_sudoku(difficulty='medium'):
    difficulty_levels = {
        'easy': 35,      # More filled cells = easier
        'medium': 25,
        'hard': 20,
        'expert': 17     # 17 is minimum for unique solution
    }
    
    filled_cells = difficulty_levels.get(difficulty, 25)  # Default to medium if invalid
    board = np.zeros((9, 9), dtype=int)
    
    # First, solve an empty board to get a complete valid solution
    temp_board = np.zeros((9, 9), dtype=int)
    solve_sudoku_heuristic(temp_board, [])  # Use heuristic solver for speed
    
    # Randomly select positions to keep
    all_positions = list(range(81))
    positions_to_keep = random.sample(all_positions, filled_cells)
    
    # Create the puzzle by keeping only selected positions
    for pos in positions_to_keep:
        row, col = divmod(pos, 9)
        board[row, col] = temp_board[row, col]
    
    return board

def draw_board(board, original_board, highlight_pos=None):
    table_html = "<table style='border-collapse: collapse; font-size: 16px;'>"
    for i in range(9):
        table_html += "<tr>"
        for j in range(9):
            num = board[i, j]
            style = "border: 1px solid black; width: 25px; height: 25px; text-align: center;"
            
            # Adding border styling
            if (i % 3 == 0 and i != 0):
                style += "border-top: 2px solid black;"
            if (j % 3 == 0 and j != 0):
                style += "border-left: 2px solid black;"
                
            # Highlight the most recently added number
            if highlight_pos and highlight_pos == (i, j):
                style += "background-color: #ffcccb;"
                
            # Show different color for original numbers
            if original_board[i, j] != 0:
                color = "#000000"  # Black for original numbers
            else:
                color = "#0000FF"  # Blue for added numbers
                
            table_html += f"<td style='{style}'><span style='color: {color}'>{num if num != 0 else ''}</span></td>"

        table_html += "</tr>"
    table_html += "</table>"
    return table_html

# App
st.set_page_config(layout="wide")
st.title("Sudoku Solver: Blind vs Heuristic Search Comparison")

# Add difficulty selector
if "difficulty" not in st.session_state:
    st.session_state.difficulty = 'medium'

# Initialize session state
if "original_board" not in st.session_state:
    difficulty = st.selectbox("Select Difficulty", 
                            ['easy', 'medium', 'hard', 'expert'],
                            index=['easy', 'medium', 'hard', 'expert'].index(st.session_state.difficulty))
    st.session_state.difficulty = difficulty
    st.session_state.original_board = generate_sudoku(difficulty)
    st.session_state.board_blind = np.copy(st.session_state.original_board)
    st.session_state.board_heuristic = np.copy(st.session_state.original_board)
    st.session_state.solved_blind = None
    st.session_state.solved_heuristic = None
    st.session_state.steps_blind = []
    st.session_state.steps_heuristic = []
    st.session_state.current_step_blind = 0
    st.session_state.current_step_heuristic = 0
    st.session_state.solving = False
    st.session_state.highlight_pos_blind = None
    st.session_state.highlight_pos_heuristic = None
    st.session_state.auto_play = False
    st.session_state.play_speed = 0.05
    st.session_state.skip_frames = 5

# Control buttons
col1, col2, col3 = st.columns([2,1,2])
with col2:
    st.markdown("### Initial Sudoku")
    st.markdown(draw_board(st.session_state.original_board, st.session_state.original_board), 
                unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Start Comparison", disabled=st.session_state.solving):
        st.session_state.solving = True
        st.session_state.auto_play = True
        
        # Reset boards and steps
        st.session_state.board_blind = np.copy(st.session_state.original_board)
        st.session_state.board_heuristic = np.copy(st.session_state.original_board)
        st.session_state.steps_blind = []
        st.session_state.steps_heuristic = []
        st.session_state.current_step_blind = 0
        st.session_state.current_step_heuristic = 0
        
        # Solve using both methods
        start_time = time.time()
        solve_sudoku_blind(st.session_state.board_blind, st.session_state.steps_blind)
        st.session_state.time_blind = time.time() - start_time
        
        start_time = time.time()
        solve_sudoku_heuristic(st.session_state.board_heuristic, st.session_state.steps_heuristic)
        st.session_state.time_heuristic = time.time() - start_time
        
        # Store runtime data immediately after solving
        if "runtime_history" not in st.session_state:
            st.session_state.runtime_history = []
        st.session_state.runtime_history.append(
            (st.session_state.time_blind, st.session_state.time_heuristic)
        )
        if len(st.session_state.runtime_history) > 10:
            st.session_state.runtime_history.pop(0)
        
        st.rerun()

with b2:
    if st.session_state.solving:
        auto_play_label = "⏸️ Pause" if st.session_state.auto_play else "▶️ Resume"
        if st.button(auto_play_label):
            st.session_state.auto_play = not st.session_state.auto_play
            st.rerun()

with b3:
    if st.button("Reset", disabled=not st.session_state.solving):
        # Add difficulty selector when resetting
        new_difficulty = st.selectbox("Select New Difficulty", 
                                    ['easy', 'medium', 'hard', 'expert'],
                                    index=['easy', 'medium', 'hard', 'expert'].index(st.session_state.difficulty))
        st.session_state.difficulty = new_difficulty
        st.session_state.original_board = generate_sudoku(new_difficulty)
        
        # Store memory results before reset
        if len(st.session_state.memory_blind) > 0 and len(st.session_state.memory_heuristic) > 0:
            total_memory_blind = sum(st.session_state.memory_blind)
            total_memory_heuristic = sum(st.session_state.memory_heuristic)
            st.session_state.memory_history_blind.append(total_memory_blind)
            st.session_state.memory_history_heuristic.append(total_memory_heuristic)
            st.session_state.total_memory_history.append((total_memory_blind, total_memory_heuristic))
        
        # Reset other states
        st.session_state.board_blind = np.copy(st.session_state.original_board)
        st.session_state.board_heuristic = np.copy(st.session_state.original_board)
        st.session_state.steps_blind = []
        st.session_state.steps_heuristic = []
        st.session_state.current_step_blind = 0
        st.session_state.current_step_heuristic = 0
        st.session_state.solving = False
        st.session_state.auto_play = False
        st.session_state.highlight_pos_blind = None
        st.session_state.highlight_pos_heuristic = None
        st.session_state.memory_blind = []
        st.session_state.memory_heuristic = []
        st.rerun()

# Speed controls (only show if solving)
if st.session_state.solving:
    speed_col1, speed_col2, speed_col3 = st.columns(3)
    
    with speed_col1:
        play_speed = st.slider("Playback Speed", 0.01, 0.5, st.session_state.play_speed, 0.01,
                              help="Control delay between steps (seconds)")
        st.session_state.play_speed = play_speed
        
    with speed_col2:
        skip_frames = st.slider("Skip Steps", 1, 50, st.session_state.skip_frames, 1,
                               help="Skip this many steps between visualizations for faster replay")
        st.session_state.skip_frames = skip_frames
        
    with speed_col3:
        if st.button("Skip to End"):
            # Complete both visualizations immediately
            st.session_state.current_step_blind = len(st.session_state.steps_blind)
            st.session_state.current_step_heuristic = len(st.session_state.steps_heuristic)
            st.session_state.board_blind = np.copy(st.session_state.original_board)
            st.session_state.board_heuristic = np.copy(st.session_state.original_board)
            
            # Reconstruct final boards
            for row, col, num in [step for step in st.session_state.steps_blind if step[2] > 0]:
                st.session_state.board_blind[row, col] = num
                
            for row, col, num in [step for step in st.session_state.steps_heuristic if step[2] > 0]:
                st.session_state.board_heuristic[row, col] = num
                
            st.session_state.auto_play = False
            st.session_state.highlight_pos_blind = None
            st.session_state.highlight_pos_heuristic = None
            st.rerun()

# Display boards side by side 
if st.session_state.solving:
    
    progress_col1, progress_col2 = st.columns(2)
    with progress_col1:
        blind_progress = min(100, 100 * st.session_state.current_step_blind / len(st.session_state.steps_blind)) if st.session_state.steps_blind else 0
        st.progress(blind_progress/100)
        
    with progress_col2:
        heuristic_progress = min(100, 100 * st.session_state.current_step_heuristic / len(st.session_state.steps_heuristic)) if st.session_state.steps_heuristic else 0
        st.progress(heuristic_progress/100)
        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Blind Search ({st.session_state.current_step_blind}/{len(st.session_state.steps_blind)} steps)")
        st.markdown(draw_board(st.session_state.board_blind, st.session_state.original_board, 
                              st.session_state.highlight_pos_blind), unsafe_allow_html=True)
        
        # Blind search stats
        if len(st.session_state.steps_blind) > 0:
            completed = st.session_state.current_step_blind >= len(st.session_state.steps_blind)
            runtime_text = f"Runtime: {st.session_state.time_blind:.4f} sec" if completed else "Solving..."

            total_memory_blind = sum(st.session_state.memory_blind)
            if len(st.session_state.memory_blind) > 0:
                st.write(f"Total Memory: {total_memory_blind:.2f} MB")

            valid_steps = [s for s in st.session_state.steps_blind if s[2] > 0]
            backtrack_steps = [s for s in st.session_state.steps_blind if s[2] == 0]
            
            stats_html = f"""
            <div style="font-size: 0.9em;">
                <p>{runtime_text}</p>
                <p>Total steps: {len(st.session_state.steps_blind)}</p>
                <p>Forward steps: {len(valid_steps)}</p>
                <p>Backtracking steps: {len(backtrack_steps)}</p>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
            
            # Show current step info if not at end
            if 0 <= st.session_state.current_step_blind - 1 < len(st.session_state.steps_blind):
                row, col, num = st.session_state.steps_blind[st.session_state.current_step_blind - 1]
                action = "Added" if num > 0 else "Removed"
                st.write(f"Last action: {action} {num if num > 0 else ''} at ({row}, {col})")

    with col2:
        st.markdown(f"### Heuristic Search ({st.session_state.current_step_heuristic}/{len(st.session_state.steps_heuristic)} steps)")
        st.markdown(draw_board(st.session_state.board_heuristic, st.session_state.original_board,
                              st.session_state.highlight_pos_heuristic), unsafe_allow_html=True)
        
        # Heuristic search stats
        if len(st.session_state.steps_heuristic) > 0:
            completed = st.session_state.current_step_heuristic >= len(st.session_state.steps_heuristic)
            runtime_text = f"Runtime: {st.session_state.time_heuristic:.4f} sec" if completed else "Solving..."

            total_memory_heuristic = sum(st.session_state.memory_heuristic)
            if len(st.session_state.memory_heuristic) > 0:
                st.write(f"Total Memory: {total_memory_heuristic:.2f} MB")
                
            valid_steps = [s for s in st.session_state.steps_heuristic if s[2] > 0]
            backtrack_steps = [s for s in st.session_state.steps_heuristic if s[2] == 0]
            
            stats_html = f"""
            <div style="font-size: 0.9em;">
                <p>{runtime_text}</p>
                <p>Total steps: {len(st.session_state.steps_heuristic)}</p>
                <p>Forward steps: {len(valid_steps)}</p>
                <p>Backtracking steps: {len(backtrack_steps)}</p>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)
            
            # Show current step info if not at end
            if 0 <= st.session_state.current_step_heuristic - 1 < len(st.session_state.steps_heuristic):
                row, col, num = st.session_state.steps_heuristic[st.session_state.current_step_heuristic - 1]
                action = "Added" if num > 0 else "Removed"
                st.write(f"Last action: {action} {num if num > 0 else ''} at ({row}, {col})")

    # Show comparison results when both are done
    if (st.session_state.current_step_blind >= len(st.session_state.steps_blind) and 
        st.session_state.current_step_heuristic >= len(st.session_state.steps_heuristic) and
        len(st.session_state.steps_blind) > 0 and len(st.session_state.steps_heuristic) > 0):
        
        st.markdown("### Comparison Results")
        
        # Prepare comparison metrics
        time_diff = st.session_state.time_blind - st.session_state.time_heuristic
        time_percent = (time_diff / st.session_state.time_blind) * 100 if st.session_state.time_blind > 0 else 0
        
        steps_diff = len(st.session_state.steps_blind) - len(st.session_state.steps_heuristic)
        steps_percent = (steps_diff / len(st.session_state.steps_blind)) * 100 if len(st.session_state.steps_blind) > 0 else 0
        
        blind_backtracks = len([s for s in st.session_state.steps_blind if s[2] == 0])
        heuristic_backtracks = len([s for s in st.session_state.steps_heuristic if s[2] == 0])
        backtrack_diff = blind_backtracks - heuristic_backtracks
        backtrack_percent = (backtrack_diff / blind_backtracks) * 100 if blind_backtracks > 0 else 0
        
        # Calculate memory differences
        total_memory_blind = sum(st.session_state.memory_blind)
        total_memory_heuristic = sum(st.session_state.memory_heuristic)
        memory_diff = total_memory_blind - total_memory_heuristic
        memory_percent = (memory_diff / total_memory_blind) * 100 if total_memory_blind > 0 else 0
        
        # Format comparison results
        faster = "Heuristic" if time_diff > 0 else "Blind"
        more_efficient = "Heuristic" if steps_diff > 0 else "Blind"
        less_backtracking = "Heuristic" if backtrack_diff > 0 else "Blind"
        less_memory = "Heuristic" if memory_diff > 0 else "Blind"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Time Efficiency:**
            - Blind: {st.session_state.time_blind:.4f} sec
            - Heuristic: {st.session_state.time_heuristic:.4f} sec
            - Difference: {abs(time_diff):.4f} sec ({abs(time_percent):.1f}%)
            - **Winner: {faster}**
            
            **Memory Efficiency:**
            - Blind: {total_memory_blind:.2f} MB
            - Heuristic: {total_memory_heuristic:.2f} MB
            - Difference: {abs(memory_diff):.2f} MB ({abs(memory_percent):.1f}%)
            - **Winner: {less_memory}**
            """)
            
        with col2:
            st.markdown(f"""
            **Solution Efficiency:**
            - Blind steps: {len(st.session_state.steps_blind)}
            - Heuristic steps: {len(st.session_state.steps_heuristic)}
            - Difference: {abs(steps_diff)} steps ({abs(steps_percent):.1f}%)
            - **Winner: {more_efficient}**
            
            **Backtracking Efficiency:**
            - Blind backtracks: {blind_backtracks}
            - Heuristic backtracks: {heuristic_backtracks}
            - Difference: {abs(backtrack_diff)} backtracks ({abs(backtrack_percent):.1f}%)
            - **Winner: {less_backtracking}**
            """)

# Auto-play logic (must be at the end)
if st.session_state.solving and st.session_state.auto_play:
    # Reset board_blind and board_heuristic before replaying steps
    if st.session_state.current_step_blind == 0:
        st.session_state.board_blind = np.copy(st.session_state.original_board)
        
    if st.session_state.current_step_heuristic == 0:
        st.session_state.board_heuristic = np.copy(st.session_state.original_board)

    # Only apply delay if we're not at the end of both solutions
    both_complete = (st.session_state.current_step_blind >= len(st.session_state.steps_blind) and 
                    st.session_state.current_step_heuristic >= len(st.session_state.steps_heuristic))
    
    if not both_complete:
        time.sleep(st.session_state.play_speed)
        
        # Update blind search visualization if not complete
        if st.session_state.current_step_blind < len(st.session_state.steps_blind):
            # Calculate how many steps to advance based on skip_frames
            target_step = min(st.session_state.current_step_blind + st.session_state.skip_frames, 
                              len(st.session_state.steps_blind))
            
            # Process all skipped steps but only visualize the last one
            for i in range(st.session_state.current_step_blind, target_step):
                row, col, num = st.session_state.steps_blind[i]
                st.session_state.board_blind[row, col] = num
                
            # Update current step and highlight position for the most recent step
            st.session_state.current_step_blind = target_step
            if target_step > 0 and target_step <= len(st.session_state.steps_blind):
                last_step = st.session_state.steps_blind[target_step-1]
                st.session_state.highlight_pos_blind = (last_step[0], last_step[1])
        else:
            st.session_state.highlight_pos_blind = None
            
        # Update heuristic search visualization if not complete
        if st.session_state.current_step_heuristic < len(st.session_state.steps_heuristic):
            # Calculate how many steps to advance based on skip_frames
            target_step = min(st.session_state.current_step_heuristic + st.session_state.skip_frames,
                             len(st.session_state.steps_heuristic))
            
            # Process all skipped steps but only visualize the last one  
            for i in range(st.session_state.current_step_heuristic, target_step):
                row, col, num = st.session_state.steps_heuristic[i]
                st.session_state.board_heuristic[row, col] = num
                
            # Update current step and highlight position for the most recent step
            st.session_state.current_step_heuristic = target_step
            if target_step > 0 and target_step <= len(st.session_state.steps_heuristic):
                last_step = st.session_state.steps_heuristic[target_step-1]
                st.session_state.highlight_pos_heuristic = (last_step[0], last_step[1])
        else:
            st.session_state.highlight_pos_heuristic = None
            
        st.rerun()
    else:
        # Both visualizations complete, turn off auto-play
        st.session_state.auto_play = False
        st.rerun()

# Modify the memory history section
if len(st.session_state.total_memory_history) > 0:
    st.markdown("### Performance History")
    
    tab1, tab2 = st.tabs(["Memory Usage", "Runtime"])
    
    with tab1:
        # Memory usage graph
        fig_memory = go.Figure()
        x_axis = list(range(1, len(st.session_state.total_memory_history) + 1))
        
        # Regular memory lines
        fig_memory.add_trace(go.Scatter(
            x=x_axis,
            y=[mem[0] for mem in st.session_state.total_memory_history],
            name="Blind Search",
            line=dict(color='blue'),
            mode='lines+markers'  # Add markers for better visibility
        ))
        
        fig_memory.add_trace(go.Scatter(
            x=x_axis,
            y=[mem[1] for mem in st.session_state.total_memory_history],
            name="Heuristic Search",
            line=dict(color='red'),
            mode='lines+markers'  # Add markers for better visibility
        ))
        
        fig_memory.update_layout(
            title=f"Memory Usage Comparison (Total {len(st.session_state.total_memory_history)} Boards)",
            xaxis_title="Board Number",
            yaxis_title="Memory Usage (MB)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_memory, use_container_width=True)
        
        # Add memory statistics
        col1, col2 = st.columns(2)
        with col1:
            avg_memory_blind = sum(mem[0] for mem in st.session_state.total_memory_history) / len(st.session_state.total_memory_history)
            max_memory_blind = max(mem[0] for mem in st.session_state.total_memory_history)
            min_memory_blind = min(mem[0] for mem in st.session_state.total_memory_history)
            st.markdown(f"""
            **Blind Search Memory Stats:**
            - Average Memory: {avg_memory_blind:.2f} MB
            - Max Memory: {max_memory_blind:.2f} MB
            - Min Memory: {min_memory_blind:.2f} MB
            """)
        
        with col2:
            avg_memory_heuristic = sum(mem[1] for mem in st.session_state.total_memory_history) / len(st.session_state.total_memory_history)
            max_memory_heuristic = max(mem[1] for mem in st.session_state.total_memory_history)
            min_memory_heuristic = min(mem[1] for mem in st.session_state.total_memory_history)
            st.markdown(f"""
            **Heuristic Search Memory Stats:**
            - Average Memory: {avg_memory_heuristic:.2f} MB
            - Max Memory: {max_memory_heuristic:.2f} MB
            - Min Memory: {min_memory_heuristic:.2f} MB
            """)
    
    with tab2:
        # Runtime graph
        if len(st.session_state.runtime_history) > 0:
            fig_runtime = go.Figure()
            x_axis = list(range(1, len(st.session_state.runtime_history) + 1))
            
            # Regular runtime lines
            fig_runtime.add_trace(go.Scatter(
                x=x_axis,
                y=[t[0] for t in st.session_state.runtime_history],
                name="Blind Search",
                line=dict(color='blue'),
                mode='lines+markers'  # Add markers for better visibility
            ))
            
            fig_runtime.add_trace(go.Scatter(
                x=x_axis,
                y=[t[1] for t in st.session_state.runtime_history],
                name="Heuristic Search",
                line=dict(color='red'),
                mode='lines+markers'  # Add markers for better visibility
            ))
            
            fig_runtime.update_layout(
                title=f"Runtime Comparison (Total {len(st.session_state.runtime_history)} Boards)",
                xaxis_title="Board Number",
                yaxis_title="Runtime (seconds)",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_runtime, use_container_width=True)
            
            # Add runtime statistics
            col1, col2 = st.columns(2)
            with col1:
                avg_runtime_blind = sum(t[0] for t in st.session_state.runtime_history) / len(st.session_state.runtime_history)
                max_runtime_blind = max(t[0] for t in st.session_state.runtime_history)
                min_runtime_blind = min(t[0] for t in st.session_state.runtime_history)
                st.markdown(f"""
                **Blind Search Runtime Stats:**
                - Average Runtime: {avg_runtime_blind:.4f} sec
                - Max Runtime: {max_runtime_blind:.4f} sec
                - Min Runtime: {min_runtime_blind:.4f} sec
                """)
            
            with col2:
                avg_runtime_heuristic = sum(t[1] for t in st.session_state.runtime_history) / len(st.session_state.runtime_history)
                max_runtime_heuristic = max(t[1] for t in st.session_state.runtime_history)
                min_runtime_heuristic = min(t[1] for t in st.session_state.runtime_history)
                st.markdown(f"""
                **Heuristic Search Runtime Stats:**
                - Average Runtime: {avg_runtime_heuristic:.4f} sec
                - Max Runtime: {max_runtime_heuristic:.4f} sec
                - Min Runtime: {min_runtime_heuristic:.4f} sec
                """)