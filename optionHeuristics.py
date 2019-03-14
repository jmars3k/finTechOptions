import urllib, time, os, re, csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
import json as js
import math
import random as rand
from bs4 import BeautifulSoup
import pickle
import requests
from matplotlib import style
from collections import Counter
#from sklearn import svm, cross_validation, neighbors
#from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from googlefinance import getQuotes
from yahoo_finance import Share

#******************************************************************************
#*************************** Functions ****************************************
#******************************************************************************

def fetchGF(googleticker):
    """

    :param googleticker:
    :return:
    """
    result = []
    returnedResult = False;
    tryCount = 0
    while not returnedResult:
        try:
            result = getQuotes("GE")
            returnedResult = True
        except:
            time.sleep(60)
            tryCount += 1

    return result

def yhGetHistory(myTicker, start, stop):
    """
    Helper function used by getMeData function

    myTicker: string ticker symbol
    start: string, format = "YYYY-M-D"
    stop: string, format = "YYYY-M-D"
    return: list of dictionaries
    """
    histList = []

    myStock = Share(myTicker)
    myRecord = myStock.get_historical(start, stop)

    for i in range(len(myRecord)):
        tempDate = myRecord[i]["Date"]
        tempOpen = float(myRecord[i]["Open"])
        tempHigh = float(myRecord[i]["High"])
        tempLow = float(myRecord[i]["Low"])
        tempClose = float(myRecord[i]["Close"])
        tempVolume = int(myRecord[i]["Volume"])
        tempAdjClose = float(myRecord[i]["Adj_Close"])

        histList.append({"Date" : tempDate, "Open" : round(tempOpen, 2), "High" : round(tempHigh, 2), "Low" : round(tempLow, 2),
                         "Close" : round(tempClose, 2), "Volume" : tempVolume, "Adj Close" : round(tempAdjClose, 2)})

    return histList

def getMeData(ticker, start, stop):
    """
    Builds a data frame out of data returned by yhGetHistory
    ticker: string stock ticker
    start: string, format = "YYYY-M-D"
    stop: string, format = "YYYY-M-D"
    return df: data frame with Date, Open, Close, Volume
    """

    colOrder = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]

    myData = yhGetHistory(ticker, start, stop)
    myDF = pd.DataFrame(myData)
    myDF = myDF[colOrder]
    myDF.set_index(["Date"], inplace = True)
    myDF.sort_index(ascending = True, inplace = True)

    return myDF

def altGetMeData(ticker, start, stop):
    """
    Works very sporadically
    :param ticker:
    :param start:
    :param stop:
    :return:
    """
    df = web.DataReader(ticker, 'yahoo', start, stop)
    return df

def getMeStats(df):
    """
    Calculates the daily price change of the underlyings price data and then calculates the mean, median and
    standard deviation of the daily price change.
    df: dataframe of stock data
    return: mean, median, standard deviation (tuple) of change and dataframe with new columns (Change, 20D_MA, 50D_MA,
            and 100D_MA)
    """
    temp = df["Close"]
    temp = temp.diff()
    df["Change"] = temp
    df["20D_MA"] = df["Close"].rolling(window = 20, min_periods = 0).mean()
    df["50D_MA"] = df["Close"].rolling(window = 50, min_periods = 0).mean()
    df["100D_MA"] = df["Close"].rolling(window = 100, min_periods = 0).mean()
    df.drop(df.index[0], inplace = True)
    mu = round(np.mean(df["Change"]), 3)
    med = round(np.median(df["Change"]), 3)
    s = round(np.std(df["Change"]), 3)

    return (mu, med, s), df

def runExcerciseTrial(myClass, myStats, priceAtEntry, numDays, numTrials):
    trialData = []
    for i in range(numTrials):
        samples = np.random.normal(myStats[0], myStats[2], numDays)
        sumSamples = round(np.sum(samples), 2)
        finalPrice = priceAtEntry + sumSamples
        trialData.append(myClass.inMoney(finalPrice))

    return round((np.sum(trialData)/numTrials), 3)

def runPriceTrial(myStats, priceAtPurchase, numDays, numTrials, optionLegs):
    """
    Simulates a single/multi leg option trade. Samples from a normal distribution with the mean and std of the
    change data of the underlying are picked.  The number of samples are given by numDays. The samples are summed and added
    to the price of the underlying when the trade was entered. The resulting price is the underlyings price at the end of the trade.
    The end price is passed to the findPL method of each leg, the results are added to find the value of the overall trade.

    myStats: mean, median, and standard deviation of "Change" of the underlying
    priceAtPurchase: underlying's price at entry into the trade
    numDays: number of days in the trade
    numTrials: number of times to run the trial
    optionLegs: a list of instantions of option objects
    return: list of option chain P/L at the end of the number of days in the trial(i.e. sums of P/Ls of each leg)
    """
    trialData = []
    for i in range(numTrials):              #iterate over the number of trials
        samples = np.random.normal(myStats[0], myStats[2], numDays)
        sumSamples = round(np.sum(samples), 2)
        finalPrice = priceAtPurchase + sumSamples

        temp = 0
        for j in range(len(optionLegs)):    #iterate over option legs
            temp += round(optionLegs[j].findPL(finalPrice), 2)

        trialData.append(temp)

    return trialData

# def plotOption(start, stop, myClass, myClass1 = None, myClass2 = None, myClass3 = None):
#     """
#     Calculate the value of an option or multi-leg option versus the price of the underlying
#
#     start: price of underlying to start analysis at
#     stop: price of underlying to stop analysis at
#     myClass: object of type of callOption or putOption
#     myClass1: optional 2nd leg
#     myClass2: optional 3rd leg
#     myClass3: optional 4th leg
#     return: two lists; first is price of underlying, second is value of option
#     """
#     xData = []
#     yData = []
#     for i in range(start, stop):
#         xData.append(i)
#         if (myClass1 is None) and (myClass2 is None) and (myClass3 is None):
#             yData.append(myClass.findPL(i))
#         elif (myClass2 is None) and (myClass3 is None):
#             yData.append(myClass.findPL(i) + myClass1.findPL(i))
#         elif (myClass3 is None):
#             yData.append(myClass.findPL(i) + myClass1.findPL(i) + myClass2.findPL(i))
#         else:
#             yData.append(myClass.findPL(i) + myClass1.findPL(i) + myClass2.findPL(i) + myClass3.findPL(i))
#
# #    plt.plot(xData, yData)
# #    plt.show()
#     return xData, yData

def plotOption(start, stop, optionLegs):
    """
    Calculate the value of an option or multi-leg option chain over a range of the underlying's price

    start: price of underlying to start analysis at
    stop: price of underlying to stop analysis at
    optionLegs: a list of instantions of option objects
    return: two lists; first is price of underlying, second is value of option
    """
    xData = []
    yData = []
    for i in np.arange(start, stop, (stop - start)/100):            #iterate over price
        xData.append(i)
        temp = 0
        for j in range(len(optionLegs)):    #iterate over option legs
            temp += optionLegs[j].findPL(i)

        yData.append(temp)

    return xData, yData

def buildSimulateChain():
    """
    The user enters data to build an option chain which is then simulated.  First each leg of the chain is simulated to
    look at the probability that it will close above the strike price for a range of the underlying's price centered
    about its price when the trade was entered.  Second the trade is simulated to show the probability of final yields
    at expiration.

    return: null, results appear as graphs
    """

    yes = ["YES", "Yes", "yes", "Y", "y"]
    no = ["NO", "No", "no", "N", "n"]
    call = ["Call", "call", "CALL", "c", "C"]
    put = ["Put", "put", "PUT", "p", "P"]
    ticker = ""

    decision = input("Do you want to analyze an option chain?  ")
    if decision in yes:
        # enter basic data
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
        while ticker == "":
            ticker = (input("Please tell me the TICKER of underlying  ")).upper()
            if ticker not in tickers:
                print("Sorry that doesn't seem to be a valid ticker symbol\n")
                ticker = ""

        priceAtEntry = float(input("Please tell me its current price? "))

        decision = input("Do you want to load new historical price data?  ")
        if decision in yes:
            myDF = getMeData(ticker, "2016-1-2", "2017-3-27")
            myDF.to_csv(ticker + "csv")
            myStats, _ = getMeStats(myDF)
        else:
            myDF = pd.read_csv(ticker + ".csv", parse_dates=True, index_col=0)
            myStats, _ = getMeStats(myDF)

        numDays = int(input("How many DAYS are left until expiration?  "))
        numTrials = int(input("How many TRIALS do you want to do for each data point?  "))

        # enter option leg data
        enterAnother = "y"
        legs = list()
        while (enterAnother in yes):
            optionType = input("What TYPE of option do you want, Put or Call?  ")
            buySell = input("Are you BUYING the options?  ")
            if buySell in yes:
                buyOption = True
            else:
                buyOption = False
            optionQty = float(input("HOW MANY options do you want?   "))
            strikePrice = round(float(input("What is the STRIKE price of the option?  ")), 2)
            purchasePrice = round(float(input("What is the PURCHASE price of the option?  ")), 2)
            fee = float(input("What is the FEE to buy or sell?  "))

            if optionType in call:
                legs.append(callOption(purchasePrice, fee, strikePrice, optionQty, buyOption))
            else:
                legs.append(putOption(purchasePrice, fee, strikePrice, optionQty, buyOption))

            enterAnother = input("Do you want to add another leg to the option chain?  ")


            # Plot the probability of each legs price being above(below) the strike price after numDays versus underlying's price
            # at entering the trade.  At each underlying price numTrials simulations are done.
        xData = np.arange((priceAtEntry - 2), (priceAtEntry + 2), ((priceAtEntry - 2) + (priceAtEntry + 2)) / 100)
        for j in range(len(legs)):
            yData = []
            for i in xData:
                yData.append(runExcerciseTrial(legs[j], myStats, i, numDays, numTrials))

            fig1, (ax1) = plt.subplots(1, 1)
            ax1.set_title("{} Option Leg {} Pct Chance of Being Excercised".format(ticker, (j + 1)))
            ax1.set_xlabel("Underlying Price at Entry")
            ax1.set_ylabel("Probability of Being Excercised")
            ax1.text((priceAtEntry - 2), 0.6, "Number of Trials {}".format(numTrials))
            ax1.text((priceAtEntry - 2), 0.55, "There were {} days until expiration".format(numDays))
            ax1.text((priceAtEntry - 2), 0.5, "Strike Price is {}".format(legs[j].strike))
            ax1.plot(xData, yData)
            plt.show()

            # Plot the yield curve for the option chain and a histogram of distribution of yields for numTtrial simulations
        someX, someY = plotOption(0.75 * priceAtEntry, 1.25 * priceAtEntry, legs)
        yields = runPriceTrial(myStats, priceAtEntry, numDays, numTrials, legs)
        num_bins = 20

        c = Counter(yields)
        mostCommon = c.most_common(3)
        top = mostCommon[0][1]  # want to find top left corner to position text in fig2
        left = np.min(yields)  #
        first = mostCommon[0][0]
        firstPercent = round(((mostCommon[0][1] / numTrials) * 100), 2)
        second = mostCommon[1][0]
        secondPercent = round(((mostCommon[1][1] / numTrials) * 100), 2)
        third = mostCommon[2][0]
        thirdPercent = round(((mostCommon[2][1] / numTrials) * 100), 2)

        fig2, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title("{} Option Chain Yield".format(ticker))
        ax1.set_xlabel("Underlying Price")
        ax1.set_ylabel("Yield")
        ax1.plot(someX, someY)
        ax2.set_title("{} Option Chain Yield Histogram".format(ticker))
        ax2.set_xlabel("Yield")
        ax2.set_ylabel("Trials at Yield")
        ax2.text(left, 0.95 * top, "Number of Trials {}".format(numTrials))
        ax2.text(left, 0.825 * top, "There were {} days until expiration".format(numDays))
        ax2.text(left, 0.70 * top, "Most common is {} at {}%".format(first, firstPercent))
        ax2.text(left, 0.575 * top, "2nd most common is {} at {}%".format(second, secondPercent))
        ax2.text(left, 0.45 * top, "3rd most common is {} at {}%".format(third, thirdPercent))
        ax2.hist(yields, num_bins)

        fig2.tight_layout(h_pad=0.5)
        plt.show()

def getOptionData(where = "J:\computationalFinance\Option Chain1.html", when = "Apr 7 2017"):
    """
    Scrapes HTML from Ameritrade using BueautifulSoup. Pulls out option tables for calls and
    puts for a given date

    where:  location of data
    when:   date of options
    return: data frame with call data, data frame with put data
    """
    try:
        page = open(where)
        soup = BeautifulSoup(page.read())
    except:
        print("Try Again, bad URL")
        return (), ()

    try:
        tableRow = soup.find(id = ("header" + when))            #find table row with id containing header and our date
    except:
        print("Try again, bad date")
        return (), ()

    tableBody = tableRow.find_parent()                          #we want the tably body

    optionTitleRow = tableBody.select(".optionTypeTitle")       #table is split into call/puts each has a CSS class optionTypeTitle
    row0is = optionTitleRow[0].get_text()

    if (row0is.strip() == 'Calls'):
        calls = optionTitleRow[0].find_parent().find_parent()   #go up two levels to find table body
        puts = optionTitleRow[1].find_parent().find_parent()
    else:
        calls = optionTitleRow[1].find_parent().find_parent()
        puts = optionTitleRow[0].find_parent().find_parent()

    callRows = calls.find_all("tr")
    putRows = puts.find_all("tr")
    callQuotes = []     #list of tupples each of which contains option quote data
    putQuotes = []

    #build callQuotes list of tupples
    for i in range(1, len(callRows)):                           #first row is just header info
        temp = callRows[i].find_all("td")
        temp2 = ()
        for j in range(len(temp)):
            quotePiece = temp[j].get_text().strip()             #get piece of individual quote
            if quotePiece != "":
                temp2 = temp2 + (quotePiece,)                   #add piece to the tupple
            else:
                continue
        callQuotes.append(temp2)

    #build putQuotes list of tupples
    for i in range(1, len(putRows)):
        temp = putRows[i].find_all("td")
        temp2 = ()
        for j in range(len(temp)):
            quotePiece = temp[j].get_text().strip()
            if quotePiece != "":
                temp2 = temp2 + (quotePiece,)
            else:
                continue
        putQuotes.append(temp2)

    tableColumns = ["Strike", "Bid", "Ask", "Last", "Change", "Vol", "Open Int"]
    calls = pd.DataFrame(index = np.arange(len(callQuotes)), columns = tableColumns)
    puts = pd.DataFrame(index = np.arange(len(callQuotes)), columns = tableColumns)

    for i in range(len(callQuotes)):
        tempRow = []
        for j in range(len(tableColumns)):
            dataItem = callQuotes[i][j].split()[0]
            dataItem = dataItem.replace(',', '')
            dataItem = dataItem.replace('--', '0')
            tempRow.append(round(float(dataItem), 2))
        calls.iloc[i] = tempRow

    for i in range(len(putQuotes)):
        tempRow = []
        for j in range(len(tableColumns)):
            dataItem = callQuotes[i][j].split()[0]
            dataItem = dataItem.replace(',', '')
            dataItem = dataItem.replace('--', '0')
            tempRow.append(round(float(dataItem), 2))
        puts.iloc[i] = tempRow

    return calls, puts

def save_sp500tickers():
    """
    Read SP500 stock symbols from Wikipedia and write them in a pickle
    dependency: lxml parser be installed
    return: list of SP500 stock symbols
    """
    tickers = []

    #changed from http to https
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers

def get_data_from_yahoo(start, stop, reload_sp500=False):
    """

    start: string, format = "YYYY-M-D"
    stop: string, format = "YYYY-M-D"
    reload_sp500: when true SP500 Tickers will be regenerated
    return: null
    """
    if reload_sp500:
        tickers = save_sp500tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in tickers[:5]:  #just do five for now
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = getMeData(ticker, start, stop)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            time.sleep(0.5)     #don't piss yahoo off
        else:
            print('Already have {}'.format(ticker))

    return

def compile_data():
    """
    Reads the data frames of individual tickers, throws out all comlumns other than adjusted close, and
    then joins into one big data frame
    return: data frame with adjusted close of all SP500 companies
    store: sp500_joined_closes.csv
    """

    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers[:5]): #just do 5 for now
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    main_df.to_csv('sp500_joined_closes.csv')
    return main_df

def visualize_data():
    """
    Calculate the correlation of every ticker to every other ticker in the master data frame and then
    display as a heatmap
    return: data frame with the correlations
    """
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    df_corr.to_csv('sp500corr.csv')

    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1.0, 1.0)
    plt.tight_layout()
    # plt.savefig("correlations.png", dpi = (300))
    plt.show()

    return df_corr


def process_data_for_labels(ticker):
    """

    :param ticker:
    :return:
    """
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    """

    :param args:
    :return:
    """
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    """

    :param ticker:
    :return:
    """
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

#******************************************************************************
#****************************** Classes ***************************************
#******************************************************************************


class callOption():
    """
    Encapsulation of a call options properties and methods.
    Things to do:
        add Greeks
        add Pricing Model
        add web data retrieval
        ???
    """

    def __init__(self, price, fee, strike, quantity, buy, covered = False):
        self.price = price
        self.fee = fee
        self.strike = strike
        self.quantity = quantity
        self.buy = buy
        self.covered = covered
        if buy:
            self.initial = -(quantity * 100 * price) - fee
        else:
            self.initial = (quantity * 100 * price) - fee

    def findPL(self, currentPrice):
        result = self.initial
        breakEven = 0
        if self.buy:
            if currentPrice >= (self.strike + 0.01):
                result = self.initial + 100 * self.quantity * (currentPrice - self.strike)
        else:   #sell
            if currentPrice >= (self.strike + 0.01):
                result = self.initial - 100 * self.quantity * (currentPrice - self.strike)
        return result

    def inMoney(self, currentPrice):
        result = 0
        if currentPrice >= (self.strike + 0.01):
            result = 1

        return result


class putOption():
    """
    Encapsulation of a put options properties and methods.
    Things to do:
        add Greeks
        add Pricing Model
        add web data retrieval
        ???
    """
    def __init__(self, price, fee, strike, quantity, buy, covered = False):
        self.price = price
        self.fee = fee
        self.strike = strike
        self.quantity = quantity
        self.buy = buy
        self.covered = covered
        if buy:
            self.initial = -(quantity * 100 * price) - fee
        else:
            self.initial = (quantity * 100 * price) - fee

    def findPL(self, currentPrice):
        result = self.initial
        breakEven = 0
        if self.buy:
            if currentPrice <= (self.strike - 0.01):
                result = self.initial + 100 * self.quantity * (self.strike - currentPrice)
        else:   #sell
            if currentPrice <= (self.strike - 0.01):
                result = self.initial - 100 * self.quantity * (self.strike - currentPrice)
        return result

    def inMoney(self, currentPrice):
        result = 0
        if currentPrice <= (self.strike - 0.01):
            result = 1

        return result

#******************************************************************************
#****************************** Sandbox ***************************************
#******************************************************************************


buildSimulateChain()



# callQuotes, putQuotes = getOptionData()
#
# print(callQuotes)
# print(putQuotes)
style.use('ggplot')
start = dt.datetime(2016, 1, 2)
stop = dt.datetime(2017, 3, 27)

#mySP500 = save_sp500tickers()
#print(mySP500)

#get_data_from_yahoo("2016-1-2", "2017-3-27")
#df = compile_data()
#df = visualize_data()
#tickers, df = process_data_for_labels("ABBV")
#_, _, df = extract_featuresets("ABBV")
#print(df.head())

try:
#    df = altGetMeData("GE", start, stop)
    df = web.DataReader("GE", "yahoo", start, stop)
except:
    df = getMeData("GE", "2016-1-2", "2017-3-27")
df.to_csv("GE.csv")
df = pd.read_csv("GE.csv", parse_dates = True, index_col = 0)

_, newDF = getMeStats(df)
#plt.xticks(rotation=70)
# plt.plot(newDF.index, newDF["Close"], newDF.index, newDF["Change"], newDF.index, newDF["50D_MA"])
# _, myLabels = plt.xticks()
# for label in myLabels:
#     label.set_rotation(45)
# plt.tight_layout()
#df.plot(subplots = True, sharex = True)

#plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
ax1.plot(newDF.index, newDF["Close"], newDF.index, newDF["50D_MA"])
ax2.plot(newDF["Change"])
_, myLabels = plt.xticks()
for label in myLabels:
    label.set_rotation(45)

plt.tight_layout()
plt.show()


#print(df.head())

moreData = fetchGF("GE")