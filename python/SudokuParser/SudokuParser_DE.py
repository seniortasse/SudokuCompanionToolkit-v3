import win32com.client as w3c
import win32clipboard
from PIL import Image

# import PIL.Image as Image
import os
import time
import csv


inputExcelPath = os.path.join(os.getcwd(), "Books")
outputImagePath = os.path.join(os.getcwd(), "image_files")
csvOutputFile = os.path.join(outputImagePath, "sudokuIndexFile.csv")


def openWorkbook(xlapp, xlfile):
    try:
        xlwb = xlapp.Workbooks.Open(
            Filename=xlfile, ReadOnly=False, IgnoreReadOnlyRecommended=True
        )
    except Exception as e:
        try:
            xlwb = xlapp.Workbooks(xlfile)
        except Exception as e:
            print(e)
            xlwb = None
    return xlwb


def getCellCoordinatesByStepNumber(inputStepNumber):
    # reduce step number by 1 to start with 0
    inputStepNumber = inputStepNumber - 1
    # get page number 0...100
    pageNumber = inputStepNumber // 4
    rowNumberOnPage = (inputStepNumber % 4) // 2
    columnNumberOnPage = (inputStepNumber % 4) % 2
    xlsPageStartRow = (28 * (pageNumber + 1)) + 1
    xlsStepStartRow = xlsPageStartRow + ((13 * rowNumberOnPage) + 2)
    xlsStepStopRow = xlsStepStartRow + 9
    xlsStepStartCol = 2 + (10 * columnNumberOnPage)
    xlsStepStopCol = 10 + (10 * columnNumberOnPage)
    return {
        "up": xlsStepStartRow,
        "down": xlsStepStopRow,
        "left": xlsStepStartCol,
        "right": xlsStepStopCol,
    }


def generateCSVHeaderRow(maxStepN):
    resultA = ["Problem ID", "Problem Name", "Book", "Level", "Answer"]
    for i in range(0, maxStepN):
        resultA.insert(len(resultA), "Step " + str(i + 1))
        resultA.insert(len(resultA), "Explanation " + str(i + 1))
    return resultA


def rgbToInt(rgb):
    colorInt = rgb[0] + (rgb[1] * 256) + (rgb[2] * 256 * 256)
    return colorInt


def saveRangeToImageFile(stepRange, outputImagePath, outputFileName):
    for _ in range(5):
        try:
            stepRange.CopyPicture(Appearance=w3c.constants.xlPrinter)
            time.sleep(0.01)  # Add a delay
            break
        except Exception as e:
            # print(f"CopyPicture failed with error: {e}")
            time.sleep(0.05)  # Wait before retrying

    clipboard_open = False
    for i in range(5):  # Retry up to 5 times
        try:
            win32clipboard.OpenClipboard()
            clipboard_open = True
            data1 = win32clipboard.GetClipboardData(win32clipboard.CF_ENHMETAFILE)
            win32clipboard.CloseClipboard()
            clipboard_open = False
            break
        except Exception as e:
            if clipboard_open:
                win32clipboard.CloseClipboard()
                clipboard_open = False
            time.sleep(0.02)  # Wait for 1 second before retrying
            if i == 4:  # If this was the last attempt
                print(f"Failed with error: {e}")

    if clipboard_open:
        win32clipboard.CloseClipboard()

    imgName = outputFileName
    tempImgName = "temp.bmp"

    writeFile = open(os.path.join(outputImagePath, tempImgName), "wb")
    writeFile.write(data1)
    writeFile.close()

    # img = Image.open(os.path.join(outputImagePath, tempImgName)).save(
    #     os.path.join(outputImagePath, imgName)
    # )

    # Open the image file
    img = Image.open(os.path.join(outputImagePath, tempImgName))
    # Calculate the new size
    new_size = (int(img.width * 0.25), int(img.height * 0.25))
    # Resize the image
    img = img.resize(new_size)
    # Decrease the color palette to 5 colors
    img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    # Save the resized image
    img.save(os.path.join(outputImagePath, imgName), "PNG", optimize=True)


print("Parsing started...")

xlsApp = w3c.gencache.EnsureDispatch("Excel.Application")
xlsApp.Visible = True

parsedFilesDict = {}

if os.path.isfile(csvOutputFile):
    with open(csvOutputFile, newline="") as f:
        reader = csv.reader(f, dialect="excel")
        for row in reader:
            exSudoku = row[0]
            parsedFilesDict[exSudoku] = 1
else:
    with open(csvOutputFile, "w", newline="") as f:
        writer = csv.writer(f, dialect="excel")
        writer.writerow(generateCSVHeaderRow(40))


dirCount = 0
maxDirCount = len(os.listdir(inputExcelPath))
for dirname in os.listdir(inputExcelPath):
    dirCount += 1
    if dirname.isnumeric():
        currentBookNum = int(dirname)
        bookDiri = os.path.join(inputExcelPath, dirname)

        fileCount = 0
        maxFileCount = len(os.listdir(bookDiri))

        for filename in os.listdir(bookDiri):
            fileCount += 1
            if (
                filename.endswith(".xls") or filename.endswith(".xlsx")
            ) and "~" not in filename:
                xlsFilei = os.path.join(bookDiri, filename)
                sudokuName = os.path.splitext(filename)[0].replace("_user_logs", "")
                problemId = "B-" + dirname + "-" + sudokuName

                if parsedFilesDict.get(problemId) != None:
                    print(
                        str(dirCount)
                        + "/"
                        + str(maxDirCount)
                        + " Book:"
                        + dirname
                        + " "
                        + str(fileCount)
                        + "/"
                        + str(maxFileCount)
                        + " File:"
                        + sudokuName
                        + " - done"
                    )
                    continue

                try:
                    xlsWB = openWorkbook(xlsApp, xlsFilei)
                    if xlsWB is not None:
                        xlsWS = xlsWB.Worksheets("Steps")
                    else:
                        print("Failed to open workbook " + xlsFilei)
                    try:
                        xlsWB.AutoSaveOn = False
                    except Exception as e1:
                        a = 1

                    maxStepText = xlsWS.Cells.Find(
                        What="Schritt",
                        LookAt=w3c.constants.xlPart,
                        SearchOrder=w3c.constants.xlByRows,
                        SearchDirection=w3c.constants.xlPrevious,
                        MatchCase=True,
                        SearchFormat=False,
                    )
                    maxStepNum = int(maxStepText.Value.split(" ")[1])

                    csvFileObj = open(csvOutputFile, "a", newline="")
                    csvWriter = csv.writer(csvFileObj, dialect="excel")

                    answerFileName = problemId + "_answer.png"
                    outputRow = [
                        problemId,
                        sudokuName,
                        "Book " + dirname,
                        "Level " + sudokuName.split("-")[1],
                        "image_files/" + answerFileName,
                    ]

                    for iStep in range(1, maxStepNum + 1):
                        cellCoord = getCellCoordinatesByStepNumber(iStep)

                        stepRange = xlsWS.Range(
                            xlsWS.Cells(cellCoord["up"], cellCoord["left"]),
                            xlsWS.Cells(cellCoord["down"], cellCoord["right"]),
                        )

                        imgName = problemId + "_step" + str(iStep) + ".png"
                        # time.sleep(0.02)
                        saveRangeToImageFile(stepRange, outputImagePath, imgName)

                        ExplanationText = xlsWS.Cells(
                            cellCoord["up"] + 10, cellCoord["left"]
                        ).Text
                        outputRow.insert(len(outputRow), "image_files/" + imgName)
                        outputRow.insert(len(outputRow), ExplanationText)

                    # clear formatting of the last step and save the answer
                    cellCoordAns = getCellCoordinatesByStepNumber(maxStepNum)
                    answerRange = xlsWS.Range(
                        xlsWS.Cells(cellCoordAns["up"] + 1, cellCoordAns["left"]),
                        xlsWS.Cells(cellCoordAns["down"], cellCoordAns["right"]),
                    )
                    answerRange.Interior.Color = rgbToInt((255, 255, 255))
                    answerRange.Font.Color = rgbToInt((0, 0, 0))
                    # time.sleep(0.05)
                    saveRangeToImageFile(answerRange, outputImagePath, answerFileName)

                    # answerRange.CopyPicture(Format=w3c.constants.xlBitmap)

                    # imgA = ImageGrab.grabclipboard()
                    # # time.sleep(0.05)
                    # imgA.save(os.path.join(outputImagePath, answerFileName))

                    xlsWB.Close(SaveChanges=False)
                    csvWriter.writerow(outputRow)
                    csvFileObj.close()

                    # time.sleep(0.05)

                    print(
                        str(dirCount)
                        + "/"
                        + str(maxDirCount)
                        + " Book:"
                        + dirname
                        + " "
                        + str(fileCount)
                        + "/"
                        + str(maxFileCount)
                        + " File:"
                        + filename
                        + " Steps:"
                        + str(maxStepNum)
                        + " - done"
                    )

                except Exception as e:
                    print(filename)
                    print(e)
                    if xlsWB is not None:
                        xlsWB.Close(SaveChanges=False)

                finally:
                    # RELEASES RESOURCES
                    xlsWB = None
                    xlsWS = None

xlsApp = None
