import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

max_y = []
out = []

streams              = [1, 2, 4, 8, 16, 32]
kernels_per_stream   = [1, 2, 4, 8, 16, 32]
arr_sizes = [10, 15, 20]

for a in arr_sizes: 
  out.append([])
  max_y.append([])
  for s in streams:
    filename = 'results_k/out_k_'+str(a)+'_'+str(s)+'.csv'
    csv = pd.read_csv(filename) 
    out[-1].append(csv)
    max_y[-1].append(max(csv["throughput"]))


fig, ax = plt.subplots(nrows=len(arr_sizes), ncols=len(streams))

for r in range(len(ax)):
  for c in range(len(ax[r])):
    p = ax[r][c]
    p.plot(out[r][c]["kernels_per_stream"], out[r][c]["throughput"], '--r', marker="o")
    m = max(max_y[r])
    p.set_ylim(0, m+0.1*m)
    p.set_xscale('log')
    p.set_xticks(kernels_per_stream)
    p.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    p.set_ylabel('Throughput MB/s', fontsize='small')
    p.set_xlabel('kernels per stream', fontsize='small')
    p.set_title('size = ' + str(2<<arr_sizes[r]) + ' bytes; streams = '+ str(streams[c]), fontsize='small')

plt.show()
