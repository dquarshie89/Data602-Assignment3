# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:36:38 2018

@author: dquarshie
"""

import requests
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import json
from bson import json_util
from pymongo import MongoClient
from fbprophet import Prophet

tradenum=0
cash = 10000000
give_cur='BTC'
rec_cur='USD'

now = dt.datetime.now()

client = MongoClient()
db = client.blotter_database
collection = db.blotter_collection
trades = db.trades

menu = ['Crypto Info', 'Trade','Show Blotter','Show P/L','Account Standings', 'Price Predictions','Quit']

blotter = pd.DataFrame(columns=[
        'Action',
        'Bought Currency',
        'Sold Currency',
        'Quantity',
        'Price per Share',
        'Trade Timestamp',
        'Money In/Out',
        'Cash']
        ) 

col_names = ['Bought Currency','Current Market Price','Position','VWAP','UPL','RPL','Total P/L']
             #,'% of Total Shares']
pl =pd.DataFrame([['BTC',0,0,0,0,0,0]] ,columns=col_names)
eth =pd.DataFrame([['ETH',0,0,0,0,0,0]] ,columns=col_names)
pl = pl.append(eth, ignore_index=True)
pl = pl.set_index('Bought Currency')


def display_menu(menu):
    print('\nMain Menu\n')
    for m in menu:
        print(str(menu.index(m) +1) + " - " + m)

def get_quote(give_cur,rec_cur):
    gdax = requests.get("https://api.gdax.com/products/"+give_cur+"-"+rec_cur+"/book")
    gdax = pd.read_json(gdax.text)

    ask = gdax.iloc[0]['asks'][0]
    bid = gdax.iloc[0]['bids'][0]
    
    price = requests.get("https://min-api.cryptocompare.com/data/pricemultifull?fsyms="+give_cur+"&tsyms="+rec_cur)
    price_data = price.json()
    
    price_data = price_data["RAW"]
    price_data = json.dumps(price_data)
    price_data = json.loads(price_data)
    
    price_data = price_data[give_cur]
    price_data = json.dumps(price_data)
    price_data = json.loads(price_data)
    
    price_data = price_data[rec_cur]
    price_data = json.dumps(price_data)
    price_data = json.loads(price_data)
    
    history_data = requests.get("https://min-api.cryptocompare.com/data/histoday?fsym="+give_cur+"&tsym="+rec_cur+"&limit=100")
    history_data = history_data.json()
    hist_df = pd.DataFrame(history_data["Data"])
    hist_df['time'] = pd.to_datetime(hist_df['time'], unit='s')
    
    maxprice = price_data["HIGH24HOUR"]
    minprice = price_data["LOW24HOUR"]
    stdprice= round(np.std(hist_df['close']),2)
    
    return (ask, bid, minprice, maxprice, stdprice, hist_df['time'] ,hist_df['close'])

def action(trade, cash, shares):
    x = get_quote(give_cur,rec_cur)
    if trade =='Buy':
        blotter.loc[tradenum] = (['Buy', give_cur, rec_cur, float(shares), float(x[1]), pd.to_datetime('now'), round(float(x[0])*float(shares),2),cash-round(float(x[0])*float(shares),2)])
    if trade == 'Sell':
        blotter.loc[tradenum] = (['Sell', give_cur, rec_cur, float(shares), float(x[1]), pd.to_datetime('now'), round(float(x[1])*float(shares),2),cash+round(float(x[1])*float(shares),2)])

def update_pl(pl, shares):
    if trade =='Buy': 
        x = get_quote(give_cur,rec_cur)
        current_qty = pl.at[give_cur,'Position']
        new_vwap = blotter.groupby('Bought Currency').apply(wavg, 'Price per Share', 'Quantity')
        pl.at[give_cur,'Position'] = current_qty + shares
        pl.at[give_cur,'VWAP'] = round(new_vwap[0],2)
        pl.at[give_cur,'UPL'] = float(x[1])*float(shares)
        pl.at[give_cur,'Current Market Price'] =float(x[1])
        pl.at[give_cur,'Total P/L']=pl.at[give_cur,'UPL']+pl.at[give_cur,'RPL']
        
        
    elif trade =='Sell': 
        x = get_quote(give_cur,rec_cur)
        current_qty = pl.at[give_cur,'Position']
        current_upl = pl.at[give_cur,'UPL']
        new_vwap = blotter.groupby('Bought Currency').apply(wavg, 'Price per Share', 'Quantity')
        pl.at[give_cur,'Position'] = current_qty - shares
        pl.at[give_cur,'VWAP'] = round(new_vwap[0],2)
        pl.at[give_cur,'RPL'] = float(x[1])*float(shares)
        pl.at[give_cur,'Current Market Price'] =float(x[1])
        pl.at[give_cur,'UPL'] = current_upl - (float(x[1])*float(shares))
        pl.at[give_cur,'Total P/L']=pl.at[give_cur,'UPL']+pl.at[give_cur,'RPL']
        
        
    return pl

def db(blotter):
    blot = blotter.to_json()                
    blot = json_util.loads(blot)
    trades.insert_one(blot)
    
    return blot, trades

def pl_pct(pl):
    pos_pcts = pl.groupby(['Bought Currency']).agg({'Position': 'sum'})/pl.agg({'Position': 'sum'})
    pos_pcts = pd.DataFrame(pos_pcts).reset_index()
    pos_pcts.columns = ['Bought Currency','% of Total Shares']
    pl_pcts = pl.groupby(['Bought Currency']).agg({'Total P/L': 'sum'})/pl.agg({'Total P/L': 'sum'})
    pl_pcts= pl_pcts.reset_index()
    pl_pcts.columns = ['Bought Currency','% of Total P/L']
    pcts = pd.merge(pos_pcts,pl_pcts,on='Bought Currency')
    return pcts
    # pos_pcts, pl_pcts
    

def get_graph(give_cur,rec_cur):  
    quote = get_quote(give_cur,rec_cur)
    plt.plot(quote[5] ,quote[6])
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price of '+give_cur+' in '+rec_cur+' over 100 Days')
    plt.show(block=False)

def mavg(give_cur,rec_cur):
    quote = get_quote(give_cur,rec_cur)
    ret = np.cumsum(quote[6].values.tolist(), dtype=float)
    c = ret[20:] = ret[20:] - ret[:-20]
    c = ret[20 - 1:] / 20
    c = pd.DataFrame(c)
    plt.plot(c,'r-')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Moving 20 Day Avg Price of '+give_cur+' in '+rec_cur)
    plt.show(block=False)   

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

def view_pl(pl,pcts):
    print("P/L")
    print(pl)
    print(pcts)
    #print()

def cash_graph(blotter):
    plt.plot(blotter['Trade Timestamp'] ,blotter['Cash'])
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time')
    plt.ylabel('Cash')
    plt.title('Cash Position')
    plt.show(block=False)
    
def pps_graph(blotter):
    fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 1]})
    ax1.set_ylabel('BTC Price per Share',fontsize=9)
    ax2.set_ylabel('ETH Price per Share',fontsize=9)
    ax1.plot(blotter[blotter['Bought Currency'] == 'BTC']['Trade Timestamp'],blotter[blotter['Bought Currency'] == 'BTC']['Price per Share'])
    ax2.plot(blotter[blotter['Bought Currency'] == 'ETH']['Trade Timestamp'],blotter[blotter['Bought Currency'] == 'ETH']['Price per Share'])
    plt.show(block=False) 
    
def predict():
    m = Prophet()
    eth_m = Prophet()
    
    #BTC
    bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20160101&end="+time.strftime("%Y%m%d"))[0]
    bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
    bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
    bitcoin_market_info.columns = ['Date','bt_Open','bt_High','bt_Low','bt_Close','bt_Volume','bt_MarketCap']
    
    
    bitcoin_market_info['ds'] = bitcoin_market_info['Date']
    bitcoin_market_info['y'] = bitcoin_market_info['bt_Close']
    
    
    #ETH
    eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20160101&end="+time.strftime("%Y%m%d"))[0]
    eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
    eth_market_info['Volume'] = eth_market_info['Volume'].astype('int64')
    eth_market_info.columns = ['Date','eth_Open','eth_High','eth_Low','eth_Close','eth_Volume','eth_MarketCap']
    
    
    eth_market_info['ds'] = eth_market_info['Date']
    eth_market_info['y'] = eth_market_info['eth_Close']
    
    #Prophet
    forecast_data = bitcoin_market_info[['ds', 'y']].copy()
    eth_data = eth_market_info[['ds', 'y']].copy()
    
    
    m.fit(forecast_data)
    eth_m.fit(eth_data)
    
    future = m.make_future_dataframe(periods=5, freq='D')
    
    
    forecast = m.predict(future)
    eth_forecast = eth_m.predict(future)
    
    btc_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    eth_pred = eth_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    
    
    btc_pred.columns = ['Date','Price Prediction','Lowest Price Prediction','Highest Price Prediction']
    eth_pred.columns = ['Date','Price Prediction','Lowest Price Prediction','Highest Price Prediction']
    
    return btc_pred, eth_pred
 
done = True   

while done:

    display_menu(menu)
    selected = int(input('\nEnter your choice [1-7]: '))
    if selected == 1:
        print('\nCrypto Info')
        give_cur = input('\nPick crypto to look up (BTC or ETH): ') 
        get_graph(give_cur,rec_cur)
        mavg(give_cur,rec_cur)
        q = get_quote(give_cur,rec_cur)
        print('\nCurrent Price: %s' %(q[0]))
        print('\nMax Price in the Last 24hrs: %s' %(q[1]))
        print('\nMin Price in the Last 24hrs: %s' %(q[2]))
        print('\nStd Dev of Price in the Last 24hrs: %s' %(q[3]))  
    
    elif selected == 2:
        
        print('\nTrade Menu')
        trade = input('Buy or Sell?: ')
        if trade == 'Buy':
            give_cur = input('\nPick your currency to buy (BTC or ETH): ')
            shares = float(input('\nEnter Quantity: '))
            get_graph(give_cur,rec_cur)
            x = get_quote(give_cur,rec_cur)
            buy_confirm = input('\nBuy %s of %s for %s at %s for %s? (Y/N): ' % (shares, give_cur, rec_cur, x[0], round(float(x[0])*float(shares),2)))
            #Check to see if the user has enough cash to buy selected number of shares at ask price
            if buy_confirm == 'Y' and float(x[1])*float(shares) > cash:
                print('\nNot enough money to buy %s \n' %(give_cur))
                print('\nTotal Cost: ')
                print(float(x[1])*float(shares))
                print('\nRemaining Cash: ')
                print(round(cash,2))
            if buy_confirm == 'Y' and float(x[1])*float(shares) <= cash:
                tradenum += 1
                #Add the buy to the blotter
                action(trade, cash, shares)
                cash = cash - blotter[blotter['Action'] == 'Buy']['Money In/Out'].sum()
                update_pl(pl, shares)
                db(blotter)
                print('\nBlotter\n')
                print(blotter)
                print('\nRemaining Cash:\n')
                print(round(cash,2))
            if buy_confirm == 'N':
                print('\nDid not buy %s' %(give_cur))
        if trade == 'Sell':
            give_cur = input('\nPick your currency to sell: ')
            #rec_cur = input('\nPick your currency to buy: ')
            shares = float(input('\nEnter Quantity: '))
            get_graph(give_cur,rec_cur)
            x = get_quote(give_cur,rec_cur)
            sell_confirm = input('\nSell %s of %s for %s at %s for %s? (Y/N): ' % (shares, give_cur, rec_cur, x[0], round(float(x[0])*float(shares),2)))
            if sell_confirm == 'Y':
                tradenum += 1
                action(trade, cash, shares)
                cash = cash + blotter[blotter['Action'] == 'Sell']['Money In/Out'].sum()
                #df_pl=initialize_pl(give_cur,rec_cur)
                update_pl(pl, shares)
                db(blotter)
                print('\nBlotter\n')
                print(blotter)
                print('\nRemaining Cash:\n')
                print(round(cash,2))
            if sell_confirm == 'N':
                print('\nDid not sell %s' %(give_cur))
            
    elif selected == 3:
        #Show blotter
        print('\nBlotter\n')
        print(blotter) 
    
    elif selected == 4:  
        pcts=pl_pct(pl)
        view_pl(pl,pcts)
        
    elif selected == 5:
        print('\nCash Position\n')
        cash_graph(blotter)
        print('\nPrice per Share\n')
        pps_graph(blotter)
        
    elif selected == 6:
        p = predict()
        print('\nBTC Price Predictions\n')
        print(p[0])
        print('\nETH Price Predictions\n')
        print(p[1])
        

    elif selected == 7:
        print('\nThanks')
        done = False
        
    elif selected >7 or selected<1:
        print('\nPlease enter a valid choice')
    
