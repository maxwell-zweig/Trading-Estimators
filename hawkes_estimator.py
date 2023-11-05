
import numpy as np
import cmath
import math 
import time
import numba as nb 
import matplotlib.pyplot as plt 
import scipy.optimize as optim
import matplotlib.pyplot as plt
from sympy import symbols, Eq
from sympy.solvers import solve 
from scipy.stats import kstest
from scipy.stats import expon
from scipy.stats import goodness_of_fit
import scipy
from statsmodels.stats.diagnostic import kstest_exponential
from statsmodels.stats.diagnostic import lilliefors


class Simulation:
    def __init__(self, dt):
        self.b = 7
        self.theta = 100
        self.n = 3
        self.v = 2
        self.alpha_mean = 0 
        self.mid_sigma = 40
        self.dt = dt

        self.k = 0.2
        self.zeta = 10
        self.sigma_alpha = 30
        self.epsilon = 5

        self.buy_rate = self.theta
        self.sell_rate = self.theta 
        self.drift = 0

        self.current_time = 0 
        self.current_price = 1000


        self.buy_rates = []
        self.sell_rates = []
        self.prices = []



    def update_drift(self, received_buy, received_sell):
        direction = np.random.randint(0, 2)
        if direction == 0:
            self.drift += -1 * self.zeta * self.drift * self.dt - self.sigma_alpha * math.sqrt(self.dt) + self.epsilon * received_buy - self.epsilon * received_sell
        else:
            self.drift += -1 * self.zeta * self.drift * self.dt + self.sigma_alpha * math.sqrt(self.dt) + self.epsilon * received_buy - self.epsilon * received_sell


    def update_orderflow(self, received_buy, received_sell):
        """
        Updates the orderflow state parameters. 
        
        Parameter received_buy: 1 if a buy was received, 0 otherwise. 
        
        Parameter received_sell: 1 if a sell was received, 0 otherwise.
        """
        self.buy_rate += self.b * (self.theta - self.buy_rate) * self.dt + self.n * received_buy + self.v * received_sell
        self.sell_rate += self.b * (self.theta - self.sell_rate) * self.dt + self.n * received_sell + self.v * received_buy

    def update_price(self):
        direction = np.random.randint(0, 2)
        if direction == 0:
            self.current_price += -1 * self.mid_sigma * math.sqrt(self.dt) + (self.alpha_mean + self.drift) * self.dt 
        else:
            self.current_price += self.mid_sigma * math.sqrt(self.dt) + (self.alpha_mean + self.drift) * self.dt 

    def order_filled(self, price, side):
        if side == "buy":
            delta = self.current_price - price
            rate = self.buy_rate
        else:
            delta = price - self.current_price
            rate = self.sell_rate
        if delta < 0:
            intensity = rate * self.dt
            hit = 0 == np.random.randint(0, int((1 / intensity)))
        else:
            intensity = rate * np.exp(-self.k * delta) * self.dt
            if intensity == 0 or math.isinf(1 / intensity) or int(1 / intensity) >= np.iinfo(np.int32).max:
                hit = False
            elif int((1 / intensity)) == 0:
                hit = True
            else:
                hit = 0 == np.random.randint(0, int((1 / intensity))) 
        return hit 

    def fit(self, dt, beta, theta, n, v):

        @nb.jit(nopython=True)
        def generate_fitting_data(dt, beta_, theta_, n_, v_):
            states = []
            order_rec = []
            buy_lambdas = []
            sell_lambdas = []
            buy_rec_times = []
            sell_rec_times = []
            trades = []
            indicators = []
            drift = 0 
            current_time = 0
            num_trades = [] 

            
            b = beta_
            theta = theta_
            n = n_
            v = v_
            alpha_mean = 0 
            mid_sigma = 40

            zeta = 100
            sigma_alpha = 1
            epsilon = 3
            lt = 10000


            print(f'Zeta Target: {zeta} Sigma Target: {sigma_alpha} Epsilon Target: {epsilon}, Theta Target: {lt}')

            buy_rate = theta
            sell_rate = theta 
            drift = lt
            nt = 0 
            while current_time <= 10:
                
                buy_intensity = dt * buy_rate
                rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                sell_intensity = dt * sell_rate
                rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                while rec_buy and rec_sell:
                    buy_intensity = dt * buy_rate
                    rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                    sell_intensity = dt * sell_rate
                    rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                rec_buy_int = 1 if rec_buy else 0 
                rec_sell_int = 1 if rec_sell else 0 


                if rec_buy and not rec_sell:
                    order_rec.append(1)
                    buy_rec_times.append(current_time)
                    buy_lambdas.append(buy_rate)
                    trades.append(current_time)
                    indicators.append(1)
                    nt += 1
                    num_trades.append(nt)
                elif rec_sell and not rec_buy:
                    order_rec.append(-1)
                    sell_rec_times.append(current_time)
                    buy_lambdas.append(buy_rate)
                    trades.append(current_time)
                    indicators.append(-1)
                elif not rec_sell and not rec_buy:
                    order_rec.append(0)
                elif rec_sell and rec_buy:
                    print("Almost surely fucked")
                states.append(drift)
                sell_lambdas.append(sell_rate)

                buy_rate += b * (theta - buy_rate) * dt + n * rec_buy_int + v * rec_sell_int
                sell_rate += b * (theta - sell_rate) * dt + n * rec_sell_int + v * rec_buy_int

                direction = np.random.randint(0, 2)
                if direction == 0:
                    drift += zeta * (lt - drift) * dt - sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                else:
                    drift += zeta * (lt - drift) * dt + sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                current_time += dt 

            return order_rec, buy_lambdas, buy_rec_times, sell_rec_times, trades, indicators, num_trades
        
        order_rec, buy_lambdas, b_r, s_r, ts, indics, num_trades = generate_fitting_data(dt, beta, theta, n, v)
        buy_lambdas = np.array(buy_lambdas)
        order_rec = np.array(order_rec)
        num_trades = np.array(num_trades)
        b_r = np.array(b_r)
        plt.plot(buy_lambdas)
        plt.show()
        plt.plot(b_r, num_trades)
        plt.show()
        plt.hist(b_r[1:] - b_r[:-1], bins=100)
        plt.show()
        # for my purposes, this test seems to be valid up to ~2000 trades / time period
        print(scipy.stats.anderson(b_r[1:] - b_r[:-1], 'expon'))


       # print(buy_lambdas[0], buy_lambdas[1], buy_lambdas[2], buy_lambdas[3], buy_lambdas[4])

       # plt.plot(np.full_like(buy_lambdas[order_rec == 1],22))

       # print(buy_lambdas[order_rec == 1].shape[0])
       # plt.plot(buy_lambdas[order_rec == 1])
       # print('bruh')
        return np.array(ts), np.array(indics)

class MLE:
    def __init__(self, time, ts, indics, influential):
        self.time_elapsed = time 

        self.trades = ts
        self.indicators = indics
       
        self.num_buys = np.count_nonzero(self.indicators == 1)
        self.num_sells = np.count_nonzero(self.indicators == -1)
        self.influential = influential

    
    def pre_calc(self):


        num_buy_orders = len(self.br)
        num_sell_orders = len(self.sr)

        buy_exponentiations_n = np.tile(self.br, (num_buy_orders, 1))
        buy_exponentiations_v = np.tile(self.sr, (num_buy_orders, 1))

        sell_exponentiations_n = np.tile(self.sr, (num_sell_orders, 1))
        sell_exponentiations_v = np.tile(self.br, (num_sell_orders, 1))

        
        br_T = np.reshape(self.br, (num_buy_orders, 1))
        sr_T = np.reshape(self.sr, (num_sell_orders, 1))

        buy_exponentiations_n = buy_exponentiations_n - br_T
        buy_exponentiations_v = buy_exponentiations_v - br_T

        sell_exponentiations_n = sell_exponentiations_n - sr_T
        sell_exponentiations_v = sell_exponentiations_v - sr_T

        buy_exponentiations_n[buy_exponentiations_n >= 0] = -float('inf')
        buy_exponentiations_v[buy_exponentiations_v >= 0] = -float('inf')
        sell_exponentiations_n[sell_exponentiations_n >= 0] = -float('inf')
        sell_exponentiations_v[sell_exponentiations_v >= 0] = -float('inf')


        buy_exponentiations_n, buy_exponentiations_v, sell_exponentiations_n, sell_exponentiations_v = np.exp(buy_exponentiations_n), np.exp(buy_exponentiations_v), np.exp(sell_exponentiations_n), np.exp(sell_exponentiations_v)

        trade_times = np.concatenate((self.br, self.sr), 0)
        
        return buy_exponentiations_n, buy_exponentiations_v, sell_exponentiations_n, sell_exponentiations_v, trade_times

    def maxim(self):

      #  @nb.jit(nopython=True)
        def log_likelihood(x, buy_exp_n, buy_exp_v, sell_exp_n, sell_exp_v, trade_times):
            beta = x[0]
            theta = x[1]
            n = x[2]
            v = x[3]

            buy_lambdas = theta + n * np.sum(buy_exp_n ** beta, axis=1) + v * np.sum(buy_exp_v ** beta, axis=1)
            sell_lambdas = theta + n * np.sum(sell_exp_n ** beta, axis=1) + v * np.sum(sell_exp_v ** beta, axis=1)

            return -1 * ( -2 * theta * 1 + ( np.sum(np.log(buy_lambdas)) + np.sum(np.log(sell_lambdas)) - ((n + v) / beta) * trade_times.shape[0] + ((n + v) / beta) * np.sum(np.exp(-beta *(1 - trade_times))) ) )
        a1, a2, a3, a4, a5 = self.pre_calc()

        x0 = [1,1,1,1]

       # nb_list = nb.typed.List
       # a1, a2, a3, a4 = nb_list(a1), nb_list(a2), nb_list(a3), nb_list(a4)
        cur_time = time.time()
        res = optim.minimize(log_likelihood, x0=x0, args=(a1, a2, a3, a4, a5), method="L-BFGS-B", bounds=((0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf) ))
        print(res)
        print(time.time() - cur_time)
        return res.x

    def maxim_batched(self):


        def log_likelihood(x):
            beta = x[0]
            theta = x[1]
            n = x[2]
            v = x[3]

            target_batch_size = 100

            bl_sub_arrays, sl_sub_arrays = [], []
            num_buy_orders = len(self.br)
            num_sell_orders = len(self.sr)


            num_buy_batches = num_buy_orders // target_batch_size + 1
            num_sell_batches = num_sell_orders // target_batch_size + 1

            print(f'Number of Buy Batches: {num_buy_batches}')
            print(f'Number of Sell Batches: {num_sell_batches}')

            for batch_idx in range(num_buy_batches):
                batch_start_idx = batch_idx * target_batch_size
                batch_end_idx = min(batch_start_idx + target_batch_size, num_buy_orders)
                num_rows_in_batch = min(target_batch_size, num_buy_orders - batch_start_idx)

                buy_exp_n = np.tile(self.br, (num_rows_in_batch, 1))
                buy_exp_v = np.tile(self.sr, (num_rows_in_batch, 1))

                br_T = np.reshape(self.br[batch_start_idx : batch_end_idx], (num_rows_in_batch, 1))

                buy_exp_n -= br_T
                buy_exp_v -= br_T

                buy_exp_n[buy_exp_n >= 0] = -float('inf')
                buy_exp_v[buy_exp_v >= 0] = -float('inf')

                buy_exp_n = np.exp(buy_exp_n)
                buy_exp_v = np.exp(buy_exp_v)

                buy_lambdas = theta + n * np.sum(buy_exp_n ** beta, axis=1) + v * np.sum(buy_exp_v ** beta, axis=1)
                bl_sub_arrays.append(buy_lambdas)
                
            for batch_idx in range(num_sell_batches):
                batch_start_idx = batch_idx * target_batch_size
                batch_end_idx = min(batch_start_idx + target_batch_size, num_sell_orders)
                num_rows_in_batch = min(target_batch_size, num_sell_orders - batch_start_idx)

                sell_exp_n = np.tile(self.sr, (num_rows_in_batch, 1))
                sell_exp_v = np.tile(self.br, (num_rows_in_batch, 1))

                sr_T = np.reshape(self.sr[batch_start_idx : batch_end_idx], (num_rows_in_batch, 1))

                sell_exp_n -= sr_T
                sell_exp_v -= sr_T

                sell_exp_n[sell_exp_n >= 0] = -float('inf')
                sell_exp_v[sell_exp_v >= 0] = -float('inf')


                sell_exp_n = np.exp(sell_exp_n)
                sell_exp_v = np.exp(sell_exp_v)


                sell_lambdas = theta + n * np.sum(sell_exp_n ** beta, axis=1) + v * np.sum(sell_exp_v ** beta, axis=1)
                sl_sub_arrays.append(sell_lambdas)

            bl, sl = np.concatenate(bl_sub_arrays, 0), np.concatenate(sl_sub_arrays, 0)
            trade_times = np.concatenate((self.br, self.sr), 0)

            return -1 * ( -2 * theta * 1 + ( np.sum(np.log(bl)) + np.sum(np.log(sl)) - ((n + v) / beta) * trade_times.shape[0] + ((n + v) / beta) * np.sum(np.exp(-beta *(1 - trade_times))) ) )

        x0 = [1,1,1,1]

        cur_time = time.time()
        res = optim.minimize(log_likelihood, x0=x0, method="L-BFGS-B", bounds=((0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf) ))
        print(res)
        print(time.time() - cur_time)
        return res.x



      #  @nb.jit(nopython=True)
    
    
      #  @nb.jit(nopython=True)
    

    def loss(self, x, trades, indicators, num_buys, num_sells, time_elapsed, influential):

        @nb.jit(nopython = True)
        def loss_shell(x, trades, indicators, num_buys, num_sells, time_elapsed, influential):
            beta = x[0]
            theta = x[1]
            n = x[2]
            v = x[3]
            #  print('evaluation...')
            #  beta, theta, n, v = 1000, 1000, 400, 20
            #  beta, theta, n, v = 130.0, 20.0, 10.0, 2.0

            bl = np.full(shape = num_buys + num_sells, fill_value=theta + 0.0001)
            sl = np.full(shape = num_buys + num_sells, fill_value=theta + 0.0001)
            #   print(bl.shape, trades.shape, indicators.shape)

            bl[0], sl[0] = theta, theta
            #  print(trades)
            for idx in range(1, len(trades)):
                if indicators[idx - 1] == 1 and influential[idx - 1] == 1:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1] - n)
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1] - v)
                elif indicators[idx -1 ] == -1 and influential[idx - 1] == 1:
                    bl[idx] =  theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1] - v)
                    sl[idx] =  theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1] - n)
                elif indicators[idx - 1] == 1 and influential[idx - 1] == 0:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1])
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1])
                elif indicators[idx - 1] == -1 and influential[idx - 1] == 0:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1])
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1])
           
            #  print(bl[0], bl[1], bl[2], bl[3], bl[4])
            #  plt.plot(bl)
            #  plt.show()
            #   print(time_elapsed)
           # print('failure')
            return -1 * (-2 * theta * time_elapsed + ( np.sum(np.log(bl[indicators == 1])) + np.sum(np.log(sl[indicators == -1])) + ((n + v) / beta) * np.sum(-1 + np.exp(-beta *(time_elapsed - trades[influential ==1])))))
        l = loss_shell(x, trades, indicators, num_buys, num_sells, time_elapsed, influential)
        return l;

    def maxim_linear(self):
        @nb.jit(nopython = True)
        def log_likelihood(x, trades, indicators, num_buys, num_sells, time_elapsed, influential):
            beta = x[0]
            theta = x[1]
            n = x[2]
            v = x[3]
            #  print('evaluation...')
            #  beta, theta, n, v = 1000, 1000, 400, 20
            #  beta, theta, n, v = 130.0, 20.0, 10.0, 2.0

            bl = np.full(shape = num_buys + num_sells, fill_value=theta + 0.0001)
            sl = np.full(shape = num_buys + num_sells, fill_value=theta + 0.0001)
            #   print(bl.shape, trades.shape, indicators.shape)

            bl[0], sl[0] = theta, theta
            #  print(trades)
            for idx in range(1, len(trades)):
                if indicators[idx - 1] == 1 and influential[idx - 1] == 1:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1] - n)
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1] - v)
                elif indicators[idx -1 ] == -1 and influential[idx - 1] == 1:
                    bl[idx] =  theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1] - v)
                    sl[idx] =  theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1] - n)
                elif indicators[idx - 1] == 1 and influential[idx - 1] == 0:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1])
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1])
                elif indicators[idx - 1] == -1 and influential[idx - 1] == 0:
                    bl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - bl[idx - 1])
                    sl[idx] = theta - np.exp(-beta * (trades[idx] - trades[idx - 1])) * (theta - sl[idx - 1])
           
            #  print(bl[0], bl[1], bl[2], bl[3], bl[4])
            #  plt.plot(bl)
            #  plt.show()
            #   print(time_elapsed)
           # print('failure')
            return -1 * (-2 * theta * time_elapsed + ( np.sum(np.log(bl[indicators == 1])) + np.sum(np.log(sl[indicators == -1])) + ((n + v) / beta) * np.sum(-1 + np.exp(-beta *(time_elapsed - trades[influential ==1])))))

        theta_init = self.trades.shape[0] / (2 * self.time_elapsed)


        x0 = [1,theta_init,1,1]

       # nb_list = nb.typed.List
       # a1, a2, a3, a4 = nb_list(a1), nb_list(a2), nb_list(a3), nb_list(a4)
        res = optim.minimize(log_likelihood, x0=x0, args=(self.trades, self.indicators, self.num_buys, self.num_sells, self.time_elapsed, self.influential), method="L-BFGS-B" , bounds=((0.000001, np.inf),(0.000001, np.inf),(0.000001, np.inf),(0.000001, np.inf)), options={'maxiter' : 100000000, 'disp' :False})
      #  res = optim.brute(log_likelihood, ranges=((4500, 5500), (9000, 10000), (50, 150), (10, 30)), Ns=15, args=(self.trades, self.indicators, self.num_buys, self.num_sells, self.time_elapsed))
       
        return res.x


if __name__ == "__main__":
    dt = 0.000001
    simulation = Simulation(dt) 
    ts, indics = simulation.fit(dt, 500, 1000, 0, 0)

    mle = MLE(10, ts, indics)
    mle.maxim_linear()
    #mle.maxim()




