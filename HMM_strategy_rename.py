import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Global constants
asset_index = 1  # Only consider BTC (the second cryptocurrency in the dataset)
T_trn = 500  # Length of the training period
split = 360  # 6-hour split for periodic updates


# Function to compute log returns over different intervals
def logRet_1(close):
    return np.array(np.diff(np.log(close)))


def logRet_2(close):
    return np.log(np.array(close[2:])) - np.log(np.array(close[:-2]))


def logRet_4(close):
    return np.log(np.array(close[2:])) - np.log(np.array(close[:-2]))


def logRet_14(close):
    return np.log(np.array(close[14:])) - np.log(np.array(close[:-14]))


def logRet_60(close):
    return np.log(np.array(close[60:])) - np.log(np.array(close[:-60]))


# Main strategy function
# This function dynamically adjusts positions and memory over time
def handle_bar(counter,  # Counter for the number of minutes tested
               time,  # Current time in string format (e.g., "2018-07-30 00:30:00")
               data,  # Market data for the current minute (in the defined format)
               init_cash,  # Initial cash balance (constant)
               transaction,  # Transaction cost ratio (constant)
               cash_balance,  # Current cash balance
               crypto_balance,  # Current cryptocurrency balance
               total_balance,  # Current total portfolio balance
               position_current,  # Current position of cryptocurrencies
               memory  # Class instance to store persistent information
               ):
    # Initialize memory variables during the first call
    if counter == 0:
        memory.avg_df = [np.mean(data[:, :3], axis=1)]  # Store the average of OHLC prices
        memory.volume = [data[:, -1]]  # Store the trading volume
        memory.vwp = np.array([]).reshape([-1, 4])  # Initialize volume-weighted price
        memory.vol_360 = np.array([]).reshape([-1, 4])  # Initialize 360-minute volume
        memory.position = np.array([0, 0, 0, 0])  # Initial portfolio position
        memory.weight = np.array([0, 0, 0, 0])  # Initial portfolio weight
        position = memory.position
    else:
        # Update average prices and volume
        avg = np.mean(data[:, :3], axis=1)
        set_balance = 45000  # Minimum portfolio balance for rebalancing
        delta = 60  # Time interval for returns calculation
        n_state = 5  # Number of hidden states for HMM
        memory.avg_df = np.append(memory.avg_df, [avg], axis=0)
        memory.volume = np.append(memory.volume, [data[:, -1]], axis=0)

        # Perform periodic updates every 6 hours
        if (counter + 1) % split == 0:
            if (sum(memory.volume) != 0).all():  # Skip if volume is zero during the period
                # Update volume-weighted prices
                memory.vwp = np.append(memory.vwp,
                                       [np.sum(
                                           memory.avg_df * memory.volume,
                                           axis=0) / np.sum(memory.volume, axis=0)], axis=0)
                memory.vol_360 = np.append(memory.vol_360, [np.sum(memory.volume, axis=0)], axis=0)

                # Train HMM if sufficient data is available
                if len(memory.vwp) >= T_trn:
                    y1 = (memory.vwp[:, 0])
                    y2 = (memory.vwp[:, 1])
                    y3 = (memory.vwp[:, 2])
                    y4 = (memory.vwp[:, 3])
                    vol1 = (memory.vol_360[:, 0])
                    vol2 = (memory.vol_360[:, 1])
                    vol3 = (memory.vol_360[:, 2])
                    vol4 = (memory.vol_360[:, 3])

                    # Initial HMM training on the first training cycle
                    if len(memory.vwp) % T_trn == 0:
                        y1_lrt_1 = logRet_1(y1)[delta - 1:]
                        y1_lrt_2 = logRet_2(y1)[delta - 2:]
                        y1_lrt_14 = logRet_14(y1)[delta - 14:]
                        y1_lrt_60 = logRet_60(y1)[max(delta - 60, 0):]

                        y2_lrt_1 = logRet_1(y2)[delta - 1:]
                        y2_lrt_2 = logRet_2(y2)[delta - 2:]
                        y2_lrt_14 = logRet_14(y2)[delta - 14:]
                        y2_lrt_60 = logRet_60(y2)[max(delta - 60, 0):]
                        y3_lrt_1 = logRet_1(y3)[delta - 1:]
                        y3_lrt_2 = logRet_2(y3)[delta - 2:]
                        y3_lrt_14 = logRet_14(y3)[delta - 14:]
                        y3_lrt_60 = logRet_60(y3)[max(delta - 60, 0):]
                        y4_lrt_1 = logRet_1(y4)[delta - 1:]
                        y4_lrt_2 = logRet_2(y4)[delta - 2:]
                        y4_lrt_14 = logRet_14(y4)[delta - 14:]
                        y4_lrt_60 = logRet_60(y4)[max(delta - 60, 0):]
                        A_y1 = np.column_stack([y1_lrt_60, y1_lrt_14, vol1[delta:]])
                        A_y2 = np.column_stack([y2_lrt_60, y2_lrt_14, vol2[delta:]])
                        A_y3 = np.column_stack([y3_lrt_60, y3_lrt_14, vol3[delta:]])
                        A_y4 = np.column_stack([y4_lrt_60, y4_lrt_14, vol4[delta:]])

                        # Fit HMM models for each cryptocurrency
                        model_y1 = hmm.GaussianHMM(n_components=n_state, covariance_type="full", n_iter=3000,
                                                   random_state=42).fit(A_y1)
                        model_y2 = hmm.GaussianHMM(n_components=n_state, covariance_type="full", n_iter=3000,
                                                   random_state=42).fit(A_y2)
                        model_y3 = hmm.GaussianHMM(n_components=n_state, covariance_type="full", n_iter=3000,
                                                   random_state=42).fit(A_y3)
                        model_y4 = hmm.GaussianHMM(n_components=n_state, covariance_type="full", n_iter=3000,
                                                   random_state=42).fit(A_y4)
                        memory.model1 = model_y1
                        memory.model2 = model_y2
                        memory.model3 = model_y3
                        memory.model4 = model_y4

                        # Predict hidden states
                        hidden_states_1 = model_y1.predict(A_y1)
                        hidden_states_2 = model_y2.predict(A_y2)
                        hidden_states_3 = model_y3.predict(A_y3)
                        hidden_states_4 = model_y4.predict(A_y4)
                        res_1 = pd.DataFrame({'logRet_1': y1_lrt_1, 'state': hidden_states_1})
                        res_2 = pd.DataFrame({'logRet_1': y2_lrt_1, 'state': hidden_states_2})
                        res_3 = pd.DataFrame({'logRet_1': y3_lrt_1, 'state': hidden_states_3})
                        res_4 = pd.DataFrame({'logRet_1': y4_lrt_1, 'state': hidden_states_4})

                        for i in range(n_state):
                            pos_y1 = (hidden_states_1 == i)
                            pos_y1 = np.append(1, pos_y1[:-1])
                            pos_y2 = (hidden_states_2 == i)
                            pos_y2 = np.append(1, pos_y2[:-1])
                            pos_y3 = (hidden_states_3 == i)
                            pos_y3 = np.append(1, pos_y3[:-1])
                            pos_y4 = (hidden_states_4 == i)
                            pos_y4 = np.append(1, pos_y4[:-1])
                            res_1['state_ret%d' % i] = (res_1.logRet_1.shift(-1).multiply(pos_y1))
                            res_2['state_ret%d' % i] = (res_2.logRet_1.shift(-1).multiply(pos_y2))
                            res_3['state_ret%d' % i] = (res_3.logRet_1.shift(-1).multiply(pos_y3))
                            res_4['state_ret%d' % i] = (res_4.logRet_1.shift(-1).multiply(pos_y4))

                        cum_res_1 = (1 + res_1.iloc[:, -n_state:].dropna()).cumprod()
                        cum_res_2 = (1 + res_2.iloc[:, -n_state:].dropna()).cumprod()
                        cum_res_3 = (1 + res_3.iloc[:, -n_state:].dropna()).cumprod()
                        cum_res_4 = (1 + res_4.iloc[:, -n_state:].dropna()).cumprod()

                        weight1, weight2, weight3, weight4 = np.zeros(n_state), np.zeros(n_state), np.zeros(
                            n_state), np.zeros(n_state)
                        weight1[np.argmax(cum_res_1.tail(1))] = 1
                        weight1[np.argmin(cum_res_1.tail(1))] = -1
                        weight2[np.argmax(cum_res_2.tail(1))] = 1
                        weight2[np.argmin(cum_res_2.tail(1))] = -1
                        weight3[np.argmax(cum_res_3.tail(1))] = 1
                        weight3[np.argmin(cum_res_3.tail(1))] = -1
                        weight4[np.argmax(cum_res_4.tail(1))] = 1
                        weight4[np.argmin(cum_res_4.tail(1))] = -1
                        memory.weight1 = weight1  # Weight corresponding to the state
                        memory.weight2 = weight2
                        memory.weight3 = weight3
                        memory.weight4 = weight4
                        if len(memory.vwp) > T_trn:
                            w = np.array([weight1[hidden_states_1[-1]],
                                          weight2[hidden_states_2[-1]],
                                          weight3[hidden_states_3[-1]],
                                          weight4[hidden_states_4[-1]]])
                            memory.weight = w
                            memory.position = memory.weight * (total_balance - set_balance) / memory.avg_df[-1,
                                                                                              :]  # rebalance
                        position = memory.position
                    else:
                        new_y1 = y1[-delta - 1:]
                        new_y2 = y2[-delta - 1:]
                        new_y3 = y3[-delta - 1:]
                        new_y4 = y4[-delta - 1:]
                        new_vol1 = vol1[-delta - 1:]
                        new_vol2 = vol2[-delta - 1:]
                        new_vol3 = vol3[-delta - 1:]
                        new_vol4 = vol4[-delta - 1:]
                        new_y1_lrt_1 = logRet_1(new_y1)[delta - 1:]
                        new_y1_lrt_2 = logRet_2(new_y1)[delta - 2:]
                        new_y1_lrt_14 = logRet_14(new_y1)[delta - 14:]
                        new_y1_lrt_60 = logRet_60(new_y1)[max(delta - 60, 0):]
                        new_y2_lrt_1 = logRet_1(new_y2)[delta - 1:]
                        new_y2_lrt_2 = logRet_2(new_y2)[delta - 2:]
                        new_y2_lrt_14 = logRet_14(new_y2)[delta - 14:]
                        new_y2_lrt_60 = logRet_60(new_y2)[max(delta - 60, 0):]
                        new_y3_lrt_1 = logRet_1(new_y3)[delta - 1:]
                        new_y3_lrt_2 = logRet_2(new_y3)[delta - 2:]
                        new_y3_lrt_14 = logRet_14(new_y3)[delta - 14:]
                        new_y3_lrt_60 = logRet_60(new_y3)[max(delta - 60, 0):]
                        new_y4_lrt_1 = logRet_1(new_y4)[delta - 1:]
                        new_y4_lrt_2 = logRet_2(new_y4)[delta - 2:]
                        new_y4_lrt_14 = logRet_14(new_y4)[delta - 14:]
                        new_y4_lrt_60 = logRet_60(new_y4)[max(delta - 60, 0):]
                        A_new_y1 = np.column_stack([new_y1_lrt_60, new_y1_lrt_14, new_vol1[delta:]])
                        A_new_y2 = np.column_stack([new_y2_lrt_60, new_y2_lrt_14, new_vol2[delta:]])
                        A_new_y3 = np.column_stack([new_y3_lrt_60, new_y3_lrt_14, new_vol3[delta:]])
                        A_new_y4 = np.column_stack([new_y4_lrt_60, new_y4_lrt_14, new_vol4[delta:]])
                        new_hidden_states_1 = memory.model1.predict(A_new_y1)
                        new_hidden_states_2 = memory.model2.predict(A_new_y2)
                        new_hidden_states_3 = memory.model3.predict(A_new_y3)
                        new_hidden_states_4 = memory.model4.predict(A_new_y4)
                        pos_y1 = memory.weight1[new_hidden_states_1[0]]
                        pos_y2 = memory.weight2[new_hidden_states_2[0]]
                        pos_y3 = memory.weight3[new_hidden_states_3[0]]
                        pos_y4 = memory.weight4[new_hidden_states_4[0]]
                        w = np.array([pos_y1, pos_y2, pos_y3, pos_y4])
                        if sum(abs(w)) != 0:
                            w /= sum(abs(w))
                        memory.weight = w
                        memory.position = memory.weight * (total_balance - set_balance) / memory.avg_df[-1,
                                                                                          :]  # rebalnace
                        position = memory.position
                else:
                    position = memory.position

            else:
                if cash_balance < 20000:
                    position = memory.weight * (min(init_cash, total_balance) - set_balance) / memory.avg_df[-1, :]
                    memory.position = [
                        position[i] if abs(position[i]) < abs([i]) else position_current[i] for i in
                        range(len(position))]
                    position = memory.position
                else:
                    memory.position = memory.weight * (total_balance - set_balance) / memory.avg_df[-1,
                                                                                      :]  # Rebalance every minute to achieve the desired weight
                    position = memory.position
            memory.avg_df = np.array([]).reshape([-1, 4])
            memory.volume = np.array([]).reshape([-1, 4])
        else:
            if cash_balance < 20000:
                position = memory.weight * (min(init_cash, total_balance) - set_balance) / memory.avg_df[-1, :]
                memory.position = [position[i] if abs(position[i]) < abs(position_current[i]) else position_current[i]
                                   for i in range(len(position))]
                position = memory.position
            else:
                memory.position = memory.weight * (total_balance - set_balance) / memory.avg_df[-1,
                                                                                  :]  # Rebalance every minute to achieve the desired weight
                position = memory.position

    return position, memory
