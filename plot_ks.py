import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

max_y = []
out = []

kernels   = [1, 2, 4, 8]
streams   = [8, 16, 32, 64]
arr_sizes = [10, 15, 20]

for a in arr_sizes: 
  out.append([])
  max_y.append([])
  for k in kernels:
    filename = 'results_ks/out_ks_'+str(a)+'_'+str(k)+'.csv'
    csv = pd.read_csv(filename) 
    out[-1].append(csv)
    max_y[-1].append(max(csv["throughput"]))


fig, ax = plt.subplots(nrows=len(arr_sizes), ncols=len(kernels))

for r in range(len(ax)):
  for c in range(len(ax[r])):
    p = ax[r][c]
    p.plot(out[r][c]["streams"], out[r][c]["throughput"], '--b', marker="o")
    m = max(max_y[r])
    p.set_ylim(0, m+0.1*m)
    p.set_xscale('log')
    p.set_xticks(streams)
    p.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    p.set_ylabel('Throughput MB/s', fontsize='small')
    p.set_xlabel('streams', fontsize='small')
    p.set_title('size = ' + str(2<<arr_sizes[r]) + ' bytes; total kernels= '+ str(kernels[c]), fontsize='small')

plt.show()
