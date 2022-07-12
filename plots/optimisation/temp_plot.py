import matplotlib.pyplot as plt


unoptimised = """Start training: qbit [1/1], r [1/7], layers [1/1]
prep
epoch [200/200] final loss 0.9738708734512329
risk = 0.9835470752086464
Training with 6 qubits, 1 layers and r=1 took 63.52408719062805s
Start training: qbit [1/1], r [2/7], layers [1/1]
prep
epoch [200/200] final loss 0.9812096953392029
risk = 0.9843781282767352
Training with 6 qubits, 1 layers and r=2 took 71.574059009552s
Start training: qbit [1/1], r [3/7], layers [1/1]
prep
epoch [200/200] final loss 0.9857870936393738
risk = 0.9806192261874669
Training with 6 qubits, 1 layers and r=4 took 98.07801413536072s
Start training: qbit [1/1], r [4/7], layers [1/1]
prep
epoch [200/200] final loss 0.9912818670272827
risk = 0.9810762509820419
Training with 6 qubits, 1 layers and r=8 took 402.0640232563019s
Start training: qbit [1/1], r [5/7], layers [1/1]               
prep                                                            
epoch [200/200] final loss 0.9929783344268799
risk = 0.9800086094787134                                         
Training with 6 qubits, 1 layers and r=16 took 1998.1272277832031s
Start training: qbit [1/1], r [6/7], layers [1/1]                 
prep                                                              
epoch [200/200] final loss 0.9929903149604797
risk = 0.9789791092130901                                        
Training with 6 qubits, 1 layers and r=32 took 8820.276495218277s
Start training: qbit [1/1], r [7/7], layers [1/1]
prep
epoch [200/200] final loss 0.9943960309028625
risk = 0.9793880313122065
Training with 6 qubits, 1 layers and r=64 took 33413.029485702515s"""


optimised = """Start training: qbit [1/1], r [1/7], layers [1/1]
prep
epoch [200/200] final loss 0.9738708734512329
risk = 0.9835470752086464                                       
Training with 6 qubits, 1 layers and r=1 took 7.551134347915649s
Start training: qbit [1/1], r [2/7], layers [1/1]               
prep
epoch [200/200] final loss 0.9812096953392029
risk = 0.9843781282767352
Training with 6 qubits, 1 layers and r=2 took 9.024479627609253s
Start training: qbit [1/1], r [3/7], layers [1/1]
prep
epoch [200/200] final loss 0.9857870936393738
risk = 0.9806192261874669
Training with 6 qubits, 1 layers and r=4 took 11.614097118377686s
Start training: qbit [1/1], r [4/7], layers [1/1]
prep
epoch [200/200] final loss 0.9912818670272827
risk = 0.9810762509820419
Training with 6 qubits, 1 layers and r=8 took 16.802953004837036s
Start training: qbit [1/1], r [5/7], layers [1/1]
prep
epoch [200/200] final loss 0.9929783344268799
risk = 0.9800086094787134
Training with 6 qubits, 1 layers and r=16 took 27.246071815490723s
Start training: qbit [1/1], r [6/7], layers [1/1]
prep
epoch [200/200] final loss 0.9929903149604797
risk = 0.9789791092130901
Training with 6 qubits, 1 layers and r=32 took 50.02910494804382s
Start training: qbit [1/1], r [7/7], layers [1/1]
prep
epoch [200/200] final loss 0.9943960309028625
risk = 0.9793880313122065
Training with 6 qubits, 1 layers and r=64 took 96.02612543106079s"""


def get_times_from_string(s):
    s = s.split('took ')[1:]
    result = [el.split('\n')[0].replace('s', '') for el in s]
    return [float(time) for time in result]


def main():
    unopt_time = get_times_from_string(unoptimised)
    opt_time = get_times_from_string(optimised)
    # r = [2**i for i in range(7)]
    r = list(range(7))
    plt.plot(r, unopt_time, label='unoptimised', marker='o')
    plt.plot(r, opt_time, label='optimised', marker='o')
    plt.xlabel('r')
    plt.ylabel('time [s]')
    plt.legend()
    plt.savefig('time_opt.png')
    plt.cla()

    factors = [unopt_time[i]/opt_time[i] for i in range(len(opt_time))]
    factors2 = [2**(i+1) for i in range(7)]
    plt.plot(r, factors, marker='o', label='achieved factor')
    plt.plot(r, factors2, marker='o', label='computed proportional factor')
    plt.xlabel('r')
    plt.ylabel('factor')
    plt.legend()
    plt.savefig('time_factor.png')
    plt.cla()


    factors_factors = [factors[i] / factors2[i] for i in range(len(factors))]
    plt.plot(r, factors_factors, marker='o')
    plt.xlabel('r')
    plt.ylabel('factors_factors')
    plt.savefig('time_factors_factors.png')
    plt.cla()

    unopt_comp_time = [2**(12+2*i) for i in range(7)]
    opt_comp_time = [2**(12+i) for i in range(7)]
    plt.plot(r, unopt_comp_time, marker='o', label='unoptimised')
    plt.plot(r, opt_comp_time, marker='o', label='optimised')
    plt.xlabel('r')
    plt.ylabel('proportional to time')
    plt.legend()
    plt.savefig('computed_times.png')
    plt.cla()


if __name__ == '__main__':
    main()
