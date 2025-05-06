We use 6h-frequency data to train our HMM model, and recognize hidden state by using observations of momentum and volume, which is 6h return, 12h return, 3.5d return, 15d return and 6h volume in detail. 
The hidden state is divided into five types, we define each state based on the cumulative returns during the training period. Recognition for ETH is shown here.
![image](https://github.com/user-attachments/assets/ea088e98-a28d-4051-ac70-a38ce9995489)
We define state with highest return the long signal, state with lowest return the short signal. Every 6h, we recognize each crypto’s current state and make decision on long or short according to the signal.
![image](https://github.com/user-attachments/assets/32538f1f-1589-4f83-9bf2-e5d9491ae52d)
To improve efficiency, we update our HMM model every 125d, using 6h data up to the training day. Thus, states may represent different signals after retraining.
Due to the lack of turnover, we can’t construct our position the moment our model gives a signal. Thus, we continuing trading until our position reaches the expected one.
Even if we achieve our expected position, we still have to rebalance every minute to guarantee weight of each asset the same.
To satisfy that we need at least 10000 cash, we require that after purchasing all the positions calculated by the model, the remaining balance must be greater than or equal to $40,000.
When our cash balance is insufficient, if of any stock at current time ‘s turnover is 0 while our model gives signal to close this position, we will cease other trades until we close this position and have enough cash balance. 
![image](https://github.com/user-attachments/assets/07e8f3fb-218c-487a-b890-47dcea6f6803)
Performance of our strategy is better than BTC’s long and short position. On the one hand, this reveal that our model has ability to recognize decreasing trend and make short decision, on the other hand, it has the ability to select profitable asset.


