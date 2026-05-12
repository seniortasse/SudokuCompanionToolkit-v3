This is a step by step guide to execute sudoku parser Python script

1) Install python libraries
2) Open window command line (CMD) and type
    pip install pypiwin32
3) Open window command line (CMD) and type
    pip install Pillow
4) Copy Sudoku *.xlsx files to “Books” folder and numeric subfolders.  Note:  The subfolder “Books”  should be located in the same folder with the SudokuParser.py script
5) Make sure that another subfolder “image_files” is created in the folder with the script
    The file and folder structure should be the following
    Folder:   “Books”
        Subfolders:
            "59"
            "60"
            "61"
            Files:
                L-1-1_user_logs.xlsx
                L-2-2_user_logs.xlsx
    …….
    Folder: “image_files”
    Script: “SudokuParser2.py”
6) Double click on  “SudokuParser2.py” to run the script
7) Image files and csv index file should be generated and saved to “image_files” folder
8) Upload files to Google folder “image_files”  (not to the root one)

IMPORTANT NOTE ' HOW TO TROUBLESHOOT AttributeError module win32com.gen_py

Delete folder: folder gen_py from my username/AppData/Local/Temp