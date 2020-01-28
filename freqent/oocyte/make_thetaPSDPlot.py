files = ['/mnt/llmStorage203/Danny/oocyte/140706_08.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/140706_09.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/140713_08.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/140717_01.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/140717_13.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/140817_05.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/160403_09.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/160403_14.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/160915_09.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/161001_04.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/161025_01.hdf5',
         '/mnt/llmStorage203/Danny/oocyte/171230_04.hdf5']


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(len(files), sharex=True)
for ind, file in enumerate(files):
    with h5py.File(file) as d:
        theta_fft = np.fft.fftshift(np.fft.fftn(d['piv']['actin']['theta']))
        theta_psd_avgk = (theta_fft * np.conj(theta_fft)).mean(axis=(1, 2))
        w = np.fft.fftshift(np.fft.fftfreq(len(theta_psd_avgk), d=d['images']['actin'].attrs['dt']))
        ax1.loglog(w[w > 0], theta_psd_avgk[w > 0])

        theta_fft = np.fft.fftn(d['piv']['actin']['theta'])
        theta_corr_avgk = np.fft.ifftn(theta_fft * np.conj(theta_fft)).mean(axis=(1, 2))
        t = np.arange(0, len(theta_corr_avgk)) * d['images']['actin'].attrs['dt']
        ax2[ind].plot(t, theta_corr_avgk, label=file.split(os.path.sep)[-1].split('.')[0])
        ax2[ind].set(xlim=[-10, 1000])
        ax2[ind].legend()
