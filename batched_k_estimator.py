
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
import torch


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

    def fit(self, dt):

        @nb.jit(nopython=True)
        def generate_fitting_data(dt):
            order_rec = []
            buy_lambdas = []
            sell_lambdas = []
            buy_rec_times = []
            sell_rec_times = []
            k_plusses = []
            k_minusses = []
            drift = 0 
            measurement_times = []
            current_time = 0 

            
            b = 500
            theta = 100
            n = 400
            v = 70
            alpha_mean = 0 
            mid_sigma = 40


            k_beta = 4000
            k_theta = 10000
            k_n = 1000
            k_v = 50


            zeta = 100
            sigma_alpha = 1
            epsilon = 3
            lt = 10000


            print(f'Theta Target: {k_theta} Beta Target: {k_beta} N Target: {k_n}, V Target: {k_v}')

            buy_rate = theta
            sell_rate = theta
            k_b = k_theta
            k_s = k_theta

            drift = lt

            while current_time <= 10:
                
                buy_intensity = dt * buy_rate
                rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                sell_intensity = dt * sell_rate
                rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                while rec_buy and rec_sell:
                    buy_intensity = dt * buy_rate
                  #  print("bruh")

                    rec_buy = 0 == np.random.randint(0, int((1 / buy_intensity))) 

                    sell_intensity = dt * sell_rate
                    rec_sell = 0 == np.random.randint(0, int((1 / sell_intensity))) 

                rec_buy_int = 1 if rec_buy else 0 
                rec_sell_int = 1 if rec_sell else 0 


                if rec_buy and not rec_sell:
                    order_rec.append(1)
                    buy_rec_times.append(current_time)
                elif rec_sell and not rec_buy:
                    order_rec.append(-1)
                    sell_rec_times.append(current_time)
                elif not rec_sell and not rec_buy:
                    order_rec.append(0)
                elif rec_sell and rec_buy:
                    print("bruh")
                k_plusses.append(k_b)
                k_minusses.append(k_s)



                measurement_times.append(current_time)

                buy_rate += b * (theta - buy_rate) * dt + n * rec_buy_int + v * rec_sell_int
                sell_rate += b * (theta - sell_rate) * dt + n * rec_sell_int + v * rec_buy_int

                k_b += k_beta * (k_theta - k_b) * dt + k_n * rec_buy + k_v * rec_sell
                k_s += k_beta * (k_theta - k_s) * dt + k_v * rec_buy + k_n * rec_sell


                direction = np.random.randint(0, 2)
                if direction == 0:
                    drift += zeta * (lt - drift) * dt - sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                else:
                    drift += zeta * (lt - drift) * dt + sigma_alpha * math.sqrt(dt) + epsilon * rec_buy_int - epsilon * rec_sell_int
                current_time += dt 
                buy_lambdas.append(buy_rate)
                sell_lambdas.append(sell_rate)


            return k_plusses, k_minusses, order_rec, measurement_times
        
        k_plusses, k_minusses, order_rec, measurement_times = generate_fitting_data(dt)
        print('done generating')
        plt.plot(k_plusses)
        plt.show()
        return np.array(k_plusses, ), np.array(k_minusses, ), np.array(order_rec, ), np.array(measurement_times)

class MLE():

    def __init__(self, k_plusses, k_minuses, order_rec, measurement_times):
        self.k_plusses = k_plusses
        self.k_minusses = k_minuses
        self.order_rec = order_rec
        self.measurement_times = measurement_times 

        self.delta_times = self.measurement_times[1:] - self.measurement_times[:-1]
        self.curtailed_order_rec = self.order_rec[:-1]
        self.curtailed_measurement_times = self.measurement_times[:-1]
        self.delta_times_squared = self.delta_times * self.delta_times
        self.num_buys = np.where(self.curtailed_order_rec == 1)[0].shape[0]
        self.num_sells = np.where(self.curtailed_order_rec == -1)[0].shape[0]

        self.num_obs = self.delta_times.shape[0] + 1
        self.kp_obs_sum = np.sum(k_plusses)
        self.km_obs_sum = np.sum(k_minuses)


        self.dsp = self.k_plusses[1:] - self.k_plusses[:-1]
        self.dsp_buy = self.dsp[np.where(self.curtailed_order_rec == 1)[0]]
        self.dsp_sell = self.dsp[np.where(self.curtailed_order_rec == -1)[0]]
        self.dsp_none = self.dsp[np.where(self.curtailed_order_rec == 0)[0]]


        self.dsp_sum = np.sum(self.dsp)
        self.dsp2_sum = np.sum(np.multiply(self.dsp, self.dsp))

        self.dsp_buy_sum = np.sum(self.dsp_buy)
        self.dsp_sell_sum = np.sum(self.dsp_sell)


        self.dsm = self.k_minusses[1:] - self.k_minusses[:-1]
        self.dsm_buy = self.dsm[np.where(self.curtailed_order_rec == 1)[0]]
        self.dsm_sell = self.dsm[np.where(self.curtailed_order_rec == -1)[0]]
        self.dsm_none = self.dsm[np.where(self.curtailed_order_rec == 0)[0]]



        self.dsm_sum = np.sum(self.dsm)
        self.dsm2_sum = np.sum(np.multiply(self.dsm, self.dsm))

        self.dsm_buy_sum = np.sum(self.dsm_buy)
        self.dsm_sell_sum = np.sum(self.dsm_sell)

        self.buy_indices = np.where(self.curtailed_order_rec == 1)[0]
        self.sell_indices = np.where(self.curtailed_order_rec == -1)[0]
        self.no_indices = np.where(self.curtailed_order_rec == 0)[0]
     #   print('done instantiating')


    def maxim(self):
        
        # length of measurmenet times = length of order rec
        def loss_simplified(x, dt, dsm_none, dsm_sell, dsm_buy, dsp_none, dsp_sell, dsp_buy, buy_indices, sell_indices, no_indices, order_rec, measurement_times):
            beta, n, v = x[0], x[1], x[2]
            
            target_batch_size = 10000
            num_rows_in_total = order_rec.shape[0]

            kp_sub_arrays = []
            km_sub_arrays = []
            num_batches = num_rows_in_total // target_batch_size + 1
            print(f'Iterating Through: {num_batches} batches')
            for batch_idx in range(num_batches):


                batch_start_idx = batch_idx * target_batch_size
                batch_end_idx = min(batch_start_idx + target_batch_size, num_rows_in_total)
                num_rows_in_batch = min(target_batch_size, num_rows_in_total - batch_start_idx)

                buy_times = measurement_times[np.where(order_rec[:batch_end_idx] == 1)[0]]
                sell_times = measurement_times[np.where(order_rec[:batch_end_idx] == -1)[0]]

                if buy_times.shape[0] > sell_times.shape[0]:
                    buy_fill = buy_times
                    sell_fill = np.full_like(buy_fill, fill_value=np.NINF)
                    sell_fill[:sell_times.shape[0]] = sell_times 

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill,(min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))

                elif buy_times.shape[0] < sell_times.shape[0]:
                    sell_fill = sell_times
                    buy_fill = np.full_like(sell_fill, fill_value=np.NINF)
                    buy_fill[:buy_times.shape[0]] = buy_times 

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    
                elif buy_times.shape[0] == sell_times.shape[0]:
                    sell_fill = sell_times
                    buy_fill = buy_times

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                
                measurement_times_T = np.reshape(measurement_times[batch_start_idx:batch_end_idx], (measurement_times[batch_start_idx:batch_end_idx].shape[0], 1))

                buy_exponentiations_n = torch.from_numpy( buy_exponentiations_n - measurement_times_T).to(device)
                buy_exponentiations_v = torch.from_numpy( buy_exponentiations_v - measurement_times_T).to(device)
                sell_exponentiations_n = torch.from_numpy(sell_exponentiations_n - measurement_times_T).to(device)
                sell_exponentiations_v = torch.from_numpy(sell_exponentiations_v - measurement_times_T).to(device)
                
                buy_exponentiations_n[buy_exponentiations_n > 0] = -float('inf')
                buy_exponentiations_v[buy_exponentiations_v > 0] = -float('inf')
                sell_exponentiations_n[sell_exponentiations_n > 0] = -float('inf')
                sell_exponentiations_v[sell_exponentiations_v > 0] = -float('inf')

                buy_exponentiations_n, buy_exponentiations_v, sell_exponentiations_n, sell_exponentiations_v = torch.exp(buy_exponentiations_n), torch.exp(buy_exponentiations_v), torch.exp(sell_exponentiations_n), torch.exp(sell_exponentiations_v)


                kp = n * torch.sum(buy_exponentiations_n ** beta, axis=1) + v * torch.sum(buy_exponentiations_n ** beta, axis=1) 
                km = n * torch.sum(sell_exponentiations_n ** beta, axis=1) + v * torch.sum(sell_exponentiations_v ** beta, axis=1) 

                kp_sub_arrays.append(kp)
                km_sub_arrays.append(km)


            kp = torch.concat(kp_sub_arrays, 0)
            km = torch.concat(km_sub_arrays, 0)

            kp_buy_portion = torch.sum((dsp_buy + beta * kp[buy_indices] * dt[buy_indices] - n) ** 2)
            kp_sell_portion = torch.sum((dsp_sell + beta * kp[sell_indices] * dt[sell_indices] - v) ** 2)
            kp_no_portion = torch.sum((dsp_none + beta * kp[no_indices] * dt[no_indices]) ** 2)

            km_buy_portion = torch.sum((dsm_buy + beta * km[buy_indices] * dt[buy_indices] - v) ** 2)
            km_sell_portion = torch.sum((dsm_sell + beta * km[sell_indices] * dt[sell_indices] - n) ** 2)
            km_no_portion = torch.sum((dsm_none + beta * km[no_indices] * dt[no_indices]) ** 2)


            return (kp_buy_portion + kp_sell_portion + kp_no_portion + km_buy_portion + km_sell_portion + km_no_portion).item()

        def theta_inference(x, kp_obs_sm, km_obs_sm, num_obs, order_rec, measurement_times):
            beta = x[0]
            n = x[1]
            v = x[2]

            target_batch_size = 10000
            num_buys = np.count_nonzero(order_rec == 1)
            num_sells = np.count_nonzero(order_rec == -1)

            num_cols = max(num_buys, num_sells)
            num_rows_in_total = order_rec.shape[0]

            kp_sub_arrays = []
            km_sub_arrays = []
            num_batches = num_rows_in_total // target_batch_size + 1
            print(f'Iterating Through: {num_batches} batches')
            for batch_idx in range(num_batches):


                batch_start_idx = batch_idx * target_batch_size
                batch_end_idx = min(batch_start_idx + target_batch_size, num_rows_in_total)
                num_rows_in_batch = min(target_batch_size, num_rows_in_total - batch_start_idx)

                print(num_rows_in_batch, batch_start_idx, batch_end_idx)

                buy_times = measurement_times[np.where(order_rec[:batch_end_idx] == 1)[0]]
                sell_times = measurement_times[np.where(order_rec[:batch_end_idx] == -1)[0]]

                if buy_times.shape[0] > sell_times.shape[0]:
                    buy_fill = buy_times
                    sell_fill = np.full_like(buy_fill, fill_value=np.NINF)
                    sell_fill[:sell_times.shape[0]] = sell_times 

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill,(min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))

                elif buy_times.shape[0] < sell_times.shape[0]:
                    sell_fill = sell_times
                    buy_fill = np.full_like(sell_fill, fill_value=np.NINF)
                    buy_fill[:buy_times.shape[0]] = buy_times 

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    
                elif buy_times.shape[0] == sell_times.shape[0]:
                    sell_fill = sell_times
                    buy_fill = buy_times

                    buy_exponentiations_n = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    buy_exponentiations_v = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_n = np.tile(sell_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                    sell_exponentiations_v = np.tile(buy_fill, (min(num_rows_in_batch, num_rows_in_total), 1))
                
                measurement_times_T = np.reshape(measurement_times[batch_start_idx:batch_end_idx], (measurement_times[batch_start_idx:batch_end_idx].shape[0], 1))

                buy_exponentiations_n = torch.from_numpy( buy_exponentiations_n - measurement_times_T).to(device)
                buy_exponentiations_v = torch.from_numpy( buy_exponentiations_v - measurement_times_T).to(device)
                sell_exponentiations_n = torch.from_numpy(sell_exponentiations_n - measurement_times_T).to(device)
                sell_exponentiations_v = torch.from_numpy(sell_exponentiations_v - measurement_times_T).to(device)
                
                buy_exponentiations_n[buy_exponentiations_n > 0] = -float('inf')
                buy_exponentiations_v[buy_exponentiations_v > 0] = -float('inf')
                sell_exponentiations_n[sell_exponentiations_n > 0] = -float('inf')
                sell_exponentiations_v[sell_exponentiations_v > 0] = -float('inf')

                buy_exponentiations_n, buy_exponentiations_v, sell_exponentiations_n, sell_exponentiations_v = torch.exp(buy_exponentiations_n), torch.exp(buy_exponentiations_v), torch.exp(sell_exponentiations_n), torch.exp(sell_exponentiations_v)


                kp = n * torch.sum(buy_exponentiations_n ** beta, axis=1) + v * torch.sum(buy_exponentiations_n ** beta, axis=1) 
                km = n * torch.sum(sell_exponentiations_n ** beta, axis=1) + v * torch.sum(sell_exponentiations_v ** beta, axis=1) 

                kp_sub_arrays.append(kp)
                km_sub_arrays.append(km)


            kp = torch.concat(kp_sub_arrays, 0)
            km = torch.concat(km_sub_arrays, 0)

            return -1 * (torch.sum(kp) + torch.sum(km) - kp_obs_sm - km_obs_sm) / (2 * num_obs)

        x0 = [1, 1, 1]
        args_tuple = (torch.from_numpy(self.delta_times).to(device), torch.from_numpy(self.dsm_none).to(device), torch.from_numpy(self.dsm_sell).to(device), torch.from_numpy(self.dsm_buy).to(device), torch.from_numpy(self.dsp_none).to(device), torch.from_numpy(self.dsp_sell).to(device), torch.from_numpy(self.dsp_buy).to(device), torch.from_numpy(self.buy_indices).to(device), torch.from_numpy(self.sell_indices).to(device), torch.from_numpy(self.no_indices).to(device), torch.from_numpy(self.curtailed_order_rec), self.curtailed_measurement_times)
        res = optim.minimize(loss_simplified, x0=x0, args=args_tuple, constraints=[{'type' : 'ineq', 'fun' : lambda x : x[0]}, {'type' : 'ineq', 'fun' : lambda x : x[1]}, {'type' : 'ineq', 'fun' : lambda x : x[2]}])
        print(res.x)

        theta_guess = theta_inference(res.x, self.kp_obs_sum, self.km_obs_sum, self.num_obs, torch.from_numpy(self.curtailed_order_rec), self.curtailed_measurement_times)
        
        print(theta_guess)

        return [res.x[0], res.x[1], res.x[2], theta_guess]

    def loss(self, x, k_plusses, k_minusses, measurement_times, indicators):
        beta, theta, n, v = x[0], x[1], x[2], x[3]
        kp, km = np.zeros(shape=measurement_times.shape[0]), np.zeros(shape=measurement_times.shape[0])

        kp[0], km[0] = theta, theta 
        for idx in range(1, len(kp)):
            if indicators[idx - 1] == 1:
                kp[idx] = theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1] - n)
                km[idx] = theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1] - v)


            elif indicators[idx - 1] == -1:
                kp[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1] - v)
                km[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1] - n)

            else:
                kp[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1])
                km[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1])

        return np.sum(((k_plusses - kp) ** 2)[:-1] * (measurement_times[1:] - measurement_times[:-1])) + np.sum(((k_minusses - km) ** 2)[:-1] * (measurement_times[1:] - measurement_times[:-1]))

    def maxim_linear(self):

        @nb.jit(nopython = True)
        def linear_loss(x, k_plusses, k_minusses, measurement_times, indicators):
            beta, theta, n, v = x[0], x[1], x[2], x[3]
            kp, km = np.zeros(shape=measurement_times.shape[0]), np.zeros(shape=measurement_times.shape[0])

            kp[0], km[0] = float(theta), float(theta) 
            for idx in range(1, len(kp)):
                if indicators[idx - 1] == 1:
                    kp[idx] = theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1] - n)
                    km[idx] = theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1] - v)


                elif indicators[idx - 1] == -1:
                    kp[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1] - v)
                    km[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1] - n)

                else:
                    kp[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - kp[idx - 1])
                    km[idx] =  theta - math.exp(-beta * (measurement_times[idx] - measurement_times[idx - 1])) * (theta - km[idx - 1])

            return np.sum(((k_plusses - kp) ** 2)[:-1] * (measurement_times[1:] - measurement_times[:-1])) + np.sum(((k_minusses - km) ** 2)[:-1] * (measurement_times[1:] - measurement_times[:-1]))
            
            
        theta_init = (np.mean(self.k_minusses) + np.mean(self.k_plusses)) / 2
        x0 = [1, theta_init, 1, 1]
        args_tuple = (self.k_plusses, self.k_minusses, self.measurement_times, self.order_rec)
        res = optim.minimize(linear_loss, x0=x0, args=args_tuple, method="L-BFGS-B", bounds=((0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf)))
       # print(res.x)


        return [res.x[0], res.x[1], res.x[2], res.x[3]]









if __name__ == "__main__":
    dt = 0.000001
    simulation = Simulation(dt) 
    kp, km, rec, mest = simulation.fit(dt)

    mle = MLE(kp, km, rec, mest)
    print(mle.maxim_linear())

