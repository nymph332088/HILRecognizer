# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:09:11 2018

@author: helihong
"""

import csv
import re
import xlwt
import xlrd
from xlutils.copy import copy    
from xlrd import open_workbook
import pandas as pd

def itemLabelForMatch(regexNum, matchString):
    matchLabels = matchString
    if regexNum == 1:
        matchLabels = 'YYYYMMDDhhmmss'
    elif regexNum == 2:
        matchLabels = 'YYYYMMDDohhommosso'
    elif regexNum == 3:
        matchLabels = 'YYYYMMDDohhommoss'
    elif regexNum == 4:
        matchLabels = 'YYYYoMMoDDohhommoss' 
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 5:
        matchLabels = 'YYYYoMMoDDohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 6:
        matchLabels = 'YYYYoMMoDDohhommoss'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 7:
        matchLabels = 'YYYYoMMoDDohhommoss'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 8:
        matchLabels = 'YYYYoMMoDDohhommosso'
    elif regexNum == 9:
        matchLabels = 'YYYYoMMoDDohhommoss'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 10:
        matchLabels = 'YYYYoMMoDDohhommoss'
    elif regexNum == 11:
        matchLabels = 'YYYYoMMoDDohhomm'
    elif regexNum == 12:
        matchLabels = 'YYYYoMMoDDohhommoss'
    elif regexNum == 13:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        space4 = matchString.index(' ', space3+1)
        matchLabels = 'M'*space1+'o'+'D'*(space2-space1-1)+'ohhommosso'+'o'*(space4-space3)+'YYYY'
    elif regexNum == 14:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        space4 = matchString.index(' ', space3+1)
        matchLabels = 'hhommo'+'o'*(space2-space1)+'D'*(space3-space2-1)+'o'+'M'*(space4-space3-1)+'oYYYY'
    elif regexNum == 15:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        matchLabels = 'hhommoo'+'D'*(space2-space1-1)+'o'+'M'*(space3-space2-1)+'oYYYY'
    elif regexNum == 16:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        space4 = matchString.index(' ', space3+1)
        matchLabels = 'M'*space1+'o'+'D'*(space2-space1-2)+'ooYYYYohhommo'+'o'*(len(matchString)-space4-1)
    elif regexNum == 17:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        space4 = matchString.index(' ', space3+1)
        space5 = matchString.index(' ', space4+1)
        matchLabels = 'M'*space1+'o'+'D'*(space2-space1-2)+'ooYYYYoooohhommo'+'o'*(len(matchString)-space5-1)
    elif regexNum == 18:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        matchLabels = 'M'*space1+'o'+'D'*(space2-space1-2)+'ooYYYYohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 19:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        matchLabels = 'M'*space1+'o'+'D'*(space2-space1-4)+'ooooYYYYoohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 20:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        space4 = matchString.index(' ', space3+1)
        matchLabels = 'D'*space1+'o'+'M'*(space2-space1-1)+'oYYYYoo'+'h'*(space4-space3-1)
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 21:
        space1 = matchString.index(' ')
        space2 = matchString.index(' ', space1+1)
        space3 = matchString.index(' ', space2+1)
        matchLabels = 'D'*space1+'o'+'M'*(space2-space1-2)+'ooYYYYohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 22:
        dash1 = matchString.index('-')
        dash2 = matchString.index('-', dash1+1)
        matchLabels = 'D'*dash1+'o'+'M'*(dash2-dash1-1)+'oYYYYohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 23:
        matchLabels = 'MMoDDoYYYYohhomm'
        matchLabels = matchLabels + 'o' * (len(matchString)-len(matchLabels))
    elif regexNum == 24:
        space1 = matchString.index(' ')
        comma1 = matchString.index(',')
        comma2 = matchString.index(',', comma1+1)
        comma3 = matchString.index(',', space1+1)
        matchLabels = 'hhommo'+'o'*(comma2-comma1)+'M'*(space1-comma2-1)+'o'+'D'*(comma3-space1-1)+'oYYYY'
    elif regexNum == 25:
        space1 = matchString.index(' ')
        comma1 = matchString.index(',')
        comma2 = matchString.index(',', comma1+1)
        matchLabels = 'o'*(comma1+1)+'M'*(space1-comma1-1)+'o'+'D'*(comma2-space1-1)+'oYYYYoohhomm'
    
    # double check the labels
    if len(matchString) == len(matchLabels): 
        return matchLabels
    else:
        return matchString

# inputList: list of outlet, kaggle time, title, url, string, prediction, character labels
def writeExcel(filename, inputList):
    # appending write
    rb = xlrd.open_workbook(filename, formatting_info=True)
    r_sheet = rb.sheet_by_index(0) 
    r = r_sheet.nrows
    wb = copy(rb) 
    sheet = wb.get_sheet(0) 
    for index, rowValue in enumerate(inputList):
        sheet.write(r+index, 0, rowValue[0])
        sheet.write(r+index, 1, rowValue[1])
        sheet.write(r+index, 2, rowValue[2])
        sheet.write(r+index, 3, rowValue[3])
        sheet.write(r+index, 4, rowValue[4])  
        sheet.write(r+index, 5, rowValue[5])  
        sheet.write(r+index, 6, rowValue[6])
    wb.save(filename)
        
#    # overwrite
#    workbook = xlwt.Workbook()
#    sheet = workbook.add_sheet('label')
#    # write headline
#    sheet.write(0, 0, 'outlet')
#    sheet.write(0, 1, 'kaggle time')
#    sheet.write(0, 2, 'title')
#    sheet.write(0, 3, 'url')
#    sheet.write(0, 4, 'string')
#    sheet.write(0, 5, 'prediction')
#    sheet.write(0, 6, 'character labels')
#    # write the kaggle time and extracted time
#    for index, rowValue in enumerate(inputList):
#        sheet.write(index+1, 0, inputList[index][0])
#        sheet.write(index+1, 1, inputList[index][1])
#        sheet.write(index+1, 2, inputList[index][2])
#        sheet.write(index+1, 3, inputList[index][3])  
#        sheet.write(index+1, 4, inputList[index][4])  
#        sheet.write(index+1, 5, inputList[index][5])  
#        sheet.write(index+1, 6, inputList[index][6])  
#    workbook.save(filename)


def main_org(): 
    csvFile = 'G:\\demo\\crawler\\experiment result\\articleTime\\testKaggleAll_Attentions.csv'
    writeCsvFile = 'G:/demo/crawler/experiment result/articleTime/testKaggleAll_CharLabels.csv'
#    writeResFile = 'G:/demo/crawler/experiment result/articleTime/testKaggleAll_CharLabels.xls'
    regexList = [r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2})([01][0-9]|2[0-3])([0-5][0-9])([0-5][0-9])', 
                 r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])Z', 
                 r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.([0-9]{1,10})Z', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.([0-9]{1,10})([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])Z',
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})-([01][0-9]|2[0-3])-([0-5][0-9])-([0-5][0-9])', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]) (\w{2,6}) (20[0-9][0-9])', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]) (\w{2,6}), (\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9])', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]), (\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9])', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) at ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9])(AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}\w{2}), (20[0-9][0-9]), ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'(\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9]), ([0-9]|0[0-9]|1[0-9]|2[0-3]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'(\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9])(AM|PM|A\.M\.|P\.M\.)', 
                 r'(\d{1,2})-((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December)-(20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'(0[1-9]|1[0-2])/(\d{1,2})/(20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (\w{2,6})', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]),(\w{3,9}),((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}),(20[0-9][0-9])', 
                 r'(\w{3,9}),((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}),(20[0-9][0-9]), ([01][0-9]|2[0-3]):([0-5][0-9])']
    writeInputList = []
    rowNum = 0
    
    '''
    newString = 'lass="fa fa-info-circle"></i> Updated: Sat,Oct 8,2017, 22:14</small></p>'
    print(newString)
    matchIdxPairList = []
    for regexIdx in range(len(regexList)):
        regex = regexList[regexIdx]
        #print(regex)
        p = re.compile(regex, re.IGNORECASE)
        iterator = p.finditer(newString)
        for match in iterator:                        
            startIdx = match.start()
            endIdx = match.end()
            activePartIndex = [regexIdx+1, startIdx, endIdx]
            matchIdxPairList.append(activePartIndex)
        # end the regex once match
        if len(matchIdxPairList) > 0:                            
            # generate labels for each character
            strLength = len(newString)
            charLabels = 'o' * strLength
            for matchIdxPair in matchIdxPairList:
                matchLabels = labelForMatch(matchIdxPair[0], newString[matchIdxPair[1]:matchIdxPair[2]])
                charLabels = charLabels[:matchIdxPair[1]] + matchLabels + charLabels[matchIdxPair[2]:]
            print(charLabels)
            break
    '''
   
    with open(csvFile, encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for csvRow in readCSV:            
            if csvRow[0] == 'Outlet':
                continue
            rowNum = rowNum + 1
#            if rowNum < 1:
#                continue
            
            outlet = None
            kaggleTime = None
            title = None
            url = None        
            newString = None
            prediction = None
            boundaryLabel = None
            charLabels = None
            if len(csvRow) >= 8:
                outlet = csvRow[0]
                kaggleTime = csvRow[1]
                title = csvRow[2]
                url = csvRow[3]
                newString = csvRow[5]
                prediction = csvRow[7]
                # apply regex for this string if predicted to 1
                if prediction == '1':
                    #print(newString)
                    matchIdxPairList = []
                    for regexIdx in range(len(regexList)):
                        regex = regexList[regexIdx]
                        #print(regex)
                        p = re.compile(regex, re.IGNORECASE)
                        iterator = p.finditer(newString)
                        for match in iterator:                        
                            startIdx = match.start()
                            endIdx = match.end()
                            activePartIndex = [regexIdx+1, startIdx, endIdx]
                            matchIdxPairList.append(activePartIndex)
                        # end the regex once match
                        if len(matchIdxPairList) > 0:                            
                            # generate labels for each character
                            strLength = len(newString)
                            charLabels = 'o' * strLength
                            boundaryLabel = 'n' * strLength
                            for matchIdxPair in matchIdxPairList:
                                matchLabels = itemLabelForMatch(matchIdxPair[0], newString[matchIdxPair[1]:matchIdxPair[2]])
                                boundaryLabel = boundaryLabel[:matchIdxPair[1]] + 'y'*(matchIdxPair[2]-matchIdxPair[1]) + boundaryLabel[matchIdxPair[2]:]
                                charLabels = charLabels[:matchIdxPair[1]] + matchLabels + charLabels[matchIdxPair[2]:]
                            #print(boundaryLabel)
                            #print(charLabels)
                            break
                    
            # keep <outlet, kaggle time, title, url, string, prediction, boundary labels, character labels> into writing list
            writeInputList.append([outlet, kaggleTime, title, url, newString, prediction, boundaryLabel, charLabels])
            
#            # write every 1000 strings, rows limited to 65535 in .xls
#            if len(writeInputList)%1000==0:
#                writeExcel(writeResFile, writeInputList)
#                writeInputList = []
#            if rowNum >= 1000:
#                break
    # write to csv file using pandas
    df = pd.DataFrame(writeInputList)
    df.to_csv(writeCsvFile, index=False)
    
def main_abnormaltest():
    abnormalFile = 'G:/demo/crawler/experiment result/attention/articleTime/abnormal.csv'
    with open(abnormalFile, encoding='utf-8') as abnormalFile:
        readCSV = csv.reader(abnormalFile, delimiter=',')
        for csvRow in readCSV:            
            if csvRow[1] == 'Outlet':
                continue
            rowNumb = csvRow[0]
            string = csvRow[5]
            boundary = csvRow[7]
            tag = csvRow[8]
            print('102: '+string[102:103])
            print('103: '+string[103:104])
            print(str(len(string)) + ' ' + str(len(boundary)) + ' '+ str(len(tag)))
            break

def readExcel(filename):
    wb = open_workbook(filename)
    for s in wb.sheets():
        values = []
        for row in range(s.nrows):
            # read each row 
            col_value = []
            for col in range(s.ncols):
                value  = (s.cell(row,col).value)
                try : value = str(int(value))
                except : pass
                col_value.append(value)
            # add to the result list
            values.append(col_value)
    # remove the headline
    values.remove(values[0])
    
    return values
        
def main():
    readFile = 'G:/demo/crawler/experiment result/attention/articleTime/testKaggle2_25.xlsx'
    writeCsvFile = 'G:/demo/crawler/experiment result/attention/articleTime/testKaggle2_25regex.csv'
    regexList = [r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2})([01][0-9]|2[0-3])([0-5][0-9])([0-5][0-9])', 
                 r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])Z', 
                 r'(20[0-9][0-9])(0[1-9]|1[0-2])(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.([0-9]{1,10})Z', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\.([0-9]{1,10})([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])Z',
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])([+-]([0-9]|0[0-9]|1[0-2]):{0,1}(00|30))', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2}) ([01][0-9]|2[0-3]):([0-5][0-9])', 
                 r'(20[0-9][0-9])-(0[1-9]|1[0-2])-(\d{2})-([01][0-9]|2[0-3])-([0-5][0-9])-([0-5][0-9])', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}) ([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]) (\w{2,6}) (20[0-9][0-9])', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]) (\w{2,6}), (\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9])', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]), (\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9])', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) at ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9])(AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}\w{2}), (20[0-9][0-9]), ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'(\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (20[0-9][0-9]), ([0-9]|0[0-9]|1[0-9]|2[0-3]) (AM|PM|A\.M\.|P\.M\.)', 
                 r'(\d{1,2}) ((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December), (20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9])(AM|PM|A\.M\.|P\.M\.)', 
                 r'(\d{1,2})-((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December)-(20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (AM|PM|A\.M\.|P\.M\.) (\w{2,6})', 
                 r'(0[1-9]|1[0-2])/(\d{1,2})/(20[0-9][0-9]) ([01][0-9]|2[0-3]):([0-5][0-9]) (\w{2,6})', 
                 r'([01][0-9]|2[0-3]):([0-5][0-9]),(\w{3,9}),((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}),(20[0-9][0-9])', 
                 r'(\w{3,9}),((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.{0,1})|January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}),(20[0-9][0-9]), ([01][0-9]|2[0-3]):([0-5][0-9])']
    writeInputList = []
    rowNum = 0        
   
    rowList = readExcel(readFile)
    if len(rowList) > 0:
        for rowNum, csvRow in enumerate(rowList):           
            outlet = None
            kaggleTime = None
            title = None
            url = None        
            newString = None
            label = None
            boundaryLabel = None
            charLabels = None
            r1 = '0'
            r2 = '0'
            r3 = '0'
            r4 = '0'
            r5 = '0'
            r6 = '0'
            r7 = '0'
            r8 = '0'
            r9 = '0'
            r10 = '0'
            r11 = '0'
            r12 = '0'
            r13 = '0'
            r14 = '0'
            r15 = '0'
            r16 = '0'
            r17 = '0'
            r18 = '0'
            r19 = '0'
            r20 = '0'
            r21 = '0'
            r22 = '0'
            r23 = '0'
            r24 = '0'
            r25 = '0'
            rtotal = None
            if len(csvRow) >= 6:
                outlet = csvRow[0]
                kaggleTime = csvRow[1]
                title = csvRow[2]
                url = csvRow[3]
                newString = csvRow[4]
                label = csvRow[5]
                # apply regex for this string if predicted to 1
                if label != '2':
                    #print(newString)
                    matchIdxPairList = []
                    for regexIdx in range(len(regexList)):
                        regex = regexList[regexIdx]
                        #print(regex)
                        p = re.compile(regex, re.IGNORECASE)
                        iterator = p.finditer(newString)
                        for match in iterator:                        
                            startIdx = match.start()
                            endIdx = match.end()
                            activePartIndex = [regexIdx+1, startIdx, endIdx]
                            matchIdxPairList.append(activePartIndex)
                        # end the regex once match
                        if len(matchIdxPairList) > 0:                            
                            # generate labels for each character
                            strLength = len(newString)
                            charLabels = 'o' * strLength
                            boundaryLabel = 'n' * strLength
                            for matchIdxPair in matchIdxPairList:
                                matchLabels = itemLabelForMatch(matchIdxPair[0], newString[matchIdxPair[1]:matchIdxPair[2]])
                                boundaryLabel = boundaryLabel[:matchIdxPair[1]] + 'y'*(matchIdxPair[2]-matchIdxPair[1]) + boundaryLabel[matchIdxPair[2]:]
                                charLabels = charLabels[:matchIdxPair[1]] + matchLabels + charLabels[matchIdxPair[2]:]
                            #print(boundaryLabel)
                            #print(charLabels)
                            
                            if regexIdx == 0:
                                r1 = '1'
                            elif regexIdx == 1:
                                r2 = '1'
                            elif regexIdx == 2:
                                r3 = '1'
                            elif regexIdx == 3:
                                r4 = '1'
                            elif regexIdx == 4:
                                r5 = '1'
                            elif regexIdx == 5:
                                r6 = '1'
                            elif regexIdx == 6:
                                r7 = '1'
                            elif regexIdx == 7:
                                r8 = '1'
                            elif regexIdx == 8:
                                r9 = '1'
                            elif regexIdx == 9:
                                r10 = '1'
                            elif regexIdx == 10:
                                r11 = '1'
                            elif regexIdx == 11:
                                r12 = '1'
                            elif regexIdx == 12:
                                r13 = '1'
                            elif regexIdx == 13:
                                r14 = '1'
                            elif regexIdx == 14:
                                r15 = '1'
                            elif regexIdx == 15:
                                r16 = '1'
                            elif regexIdx == 16:
                                r17 = '1'
                            elif regexIdx == 17:
                                r18 = '1'
                            elif regexIdx == 18:
                                r19 = '1'
                            elif regexIdx == 19:
                                r20 = '1'
                            elif regexIdx == 20:
                                r21 = '1'
                            elif regexIdx == 21:
                                r22 = '1'
                            elif regexIdx == 22:
                                r23 = '1'
                            elif regexIdx == 23:
                                r24 = '1'
                            elif regexIdx == 24:
                                r25 = '1'
                                
                            break
                    
            # keep <outlet, kaggle time, title, url, string, prediction, boundary labels, character labels> into writing list
            writeInputList.append([outlet, kaggleTime, title, url, newString, label, boundaryLabel, charLabels, 
                                   r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, 
                                   r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, 
                                   r21, r22, r23, r24, r25, rtotal])
            

    # write to csv file using pandas
    df = pd.DataFrame(writeInputList)
    df.to_csv(writeCsvFile, index=False, encoding='utf-8')
            
if __name__=='__main__':
    main() 