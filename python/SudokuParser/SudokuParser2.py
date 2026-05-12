import win32com.client as w3c
from PIL import ImageGrab
import os
import time
import csv


inputExcelPath = os.path.join(os.getcwd(),"Books")
outputImagePath = os.path.join(os.getcwd(),"image_files")
csvOutputFile = os.path.join(outputImagePath,"sudokuIndexFile.csv")

def openWorkbook(xlapp, xlfile):
    try:        
        xlwb = xlapp.Workbooks.Open(Filename=xlfile, ReadOnly=False, IgnoreReadOnlyRecommended=True)            
    except Exception as e:
        try:
            xlwb = xlapp.Workbooks(xlfile)
        except Exception as e:
            print(e)
            xlwb = None                    
    return(xlwb)

def getCellCoordinatesByStepNumber(inputStepNumber):
    #reduce step number by 1 to start with 0
    inputStepNumber = inputStepNumber - 1
    #get page number 0...100
    pageNumber = inputStepNumber // 4
    rowNumberOnPage = (inputStepNumber % 4) // 2
    columnNumberOnPage = (inputStepNumber % 4) % 2
    xlsPageStartRow = (28 * (pageNumber + 1) ) + 1
    xlsStepStartRow = xlsPageStartRow + ((13 * rowNumberOnPage) + 2)
    xlsStepStopRow = xlsStepStartRow + 9
    xlsStepStartCol = 2 + (10 * columnNumberOnPage)
    xlsStepStopCol = 10 + (10 * columnNumberOnPage)
    return {"up": xlsStepStartRow, "down": xlsStepStopRow, "left": xlsStepStartCol,  "right": xlsStepStopCol}

def generateCSVHeaderRow(maxStepN):
    resultA = ["Problem ID","Problem Name","Book","Level","Answer"]
    for i in range(0,maxStepN):
        resultA.insert(len(resultA) , "Step " + str(i+1))
        resultA.insert(len(resultA) , "Explanation " + str(i+1))
    return resultA

def rgbToInt(rgb):
    colorInt = rgb[0] + (rgb[1] * 256) + (rgb[2] * 256 * 256)
    return colorInt
     
print("Parsing started...")

xlsApp = w3c.gencache.EnsureDispatch('Excel.Application')
xlsApp.Visible = True

parsedFilesDict = {}

if(os.path.isfile(csvOutputFile)): 
    with open(csvOutputFile, newline='') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            exSudoku = row[0]
            parsedFilesDict[exSudoku] = 1
else:
    with open(csvOutputFile, 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(generateCSVHeaderRow(40))


dirCount = 0
maxDirCount =  len(os.listdir(inputExcelPath))
for dirname in os.listdir(inputExcelPath):
    dirCount += 1
    if(dirname.isnumeric()):
        currentBookNum = int(dirname)
        bookDiri =  os.path.join(inputExcelPath, dirname)

        fileCount = 0
        maxFileCount =  len(os.listdir(bookDiri))

        for filename in os.listdir(bookDiri):
            fileCount += 1
            if filename.endswith('.xls') or filename.endswith('.xlsx'):
                xlsFilei = os.path.join(bookDiri, filename)
                sudokuName = os.path.splitext(filename)[0].replace("_user_logs", "")
                problemId = "B-"+dirname+"-"+sudokuName

                if(parsedFilesDict.get(problemId)!=None):
                    print(str(dirCount)+"/"+str(maxDirCount)+" Book:"+dirname+" "+str(fileCount)+"/"+str(maxFileCount)+" File:"+sudokuName+  " - done")
                    continue


                try:
                    xlsWB = openWorkbook(xlsApp, xlsFilei)
                    xlsWS = xlsWB.Worksheets('Steps') 
                    try:
                        xlsWB.AutoSaveOn = False
                    except Exception as e1:
                        a = 1

                    maxStepText = xlsWS.Cells.Find(What="STEP", LookAt=w3c.constants.xlPart, SearchOrder=w3c.constants.xlByRows, SearchDirection=w3c.constants.xlPrevious, MatchCase=True,SearchFormat=False)
                    maxStepNum = int(maxStepText.Value.split(" ")[1])

                    csvFileObj = open(csvOutputFile, 'a', newline='')
                    csvWriter = csv.writer(csvFileObj, dialect='excel')
                    
                    answerFileName = problemId + "_answer.png"
                    outputRow = [problemId,sudokuName,"Book "+dirname,"Level "+sudokuName.split("-")[1],"image_files/"+answerFileName]
                    
                    for iStep in range(1,maxStepNum+1):
                        cellCoord = getCellCoordinatesByStepNumber(iStep)

                        xlsWS.Range(xlsWS.Cells(cellCoord["up"],cellCoord["left"]),xlsWS.Cells(cellCoord["down"],cellCoord["right"])).CopyPicture(Format= w3c.constants.xlBitmap)  
                        time.sleep(0.09)
                        img = ImageGrab.grabclipboard()
                        imgName = problemId + "_step"+ str(iStep) +".png"
                        #time.sleep(0.05)
                        img.save(os.path.join(outputImagePath,imgName))
                        
                        ExplanationText = xlsWS.Cells(cellCoord["up"]+10,cellCoord["left"]).Text                
                        outputRow.insert(len(outputRow), "image_files/"+imgName)
                        outputRow.insert(len(outputRow), ExplanationText)    
                    
                    #clear formatting of the last step and save the answer
                    cellCoordAns = getCellCoordinatesByStepNumber(maxStepNum)
                    answerRange = xlsWS.Range(xlsWS.Cells(cellCoordAns["up"]+1,cellCoordAns["left"]),xlsWS.Cells(cellCoordAns["down"],cellCoordAns["right"]))
                    answerRange.Interior.Color = rgbToInt((255,255,255))
                    answerRange.Font.Color = rgbToInt((0,0,0))
                    answerRange.CopyPicture(Format= w3c.constants.xlBitmap)
                    time.sleep(0.1)
                    imgA = ImageGrab.grabclipboard()
                    #time.sleep(0.05)
                    imgA.save(os.path.join(outputImagePath,answerFileName))

                    
                    xlsWB.Close(SaveChanges=False)
                    csvWriter.writerow(outputRow)
                    csvFileObj.close()

                    time.sleep(0.05)
                    
                    print(str(dirCount)+"/"+str(maxDirCount)+" Book:"+dirname+" "+str(fileCount)+"/"+str(maxFileCount)+" File:"+filename+ " Steps:"+ str(maxStepNum)+ " - done")

                except Exception as e:
                    
                    print(filename)
                    print(e)
                    xlsWB.Close(SaveChanges=False)

                finally:
                    # RELEASES RESOURCES
                    xlsWB = None
                    xlsWS = None

xlsApp = None


