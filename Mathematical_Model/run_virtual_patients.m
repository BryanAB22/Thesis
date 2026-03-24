
% Top 50 | least to greatest
sens_pars_50 = [ ...
     32  ... % hm10
      18  ... % xl06
      5  ... % k8
     41  ... % k6tnf
         107  ... % hmD
          53  ... % xt6
          92  ... % xI10
     15  ... % x6tnf
          36  ... % s10
     65  ... % kpepp
    113  ... % ktnfhr
         98  ... % alpha
     66  ... % kpp
          25  ... % h6tnf
      63  ... % hht
     61  ... % kh
     43  ... % k106
       50 ... % kt6
     26  ... % h66
     96  ... % mud
     48  ... % kt
          95  ... % kdn
          97  ... % xdn
           2  ... % k10m
      76  ... % knom
      77  ... % kno
     11  ... % kmr
      7  ... % ktnf
            8  ... % ktnfm
     78  ... % xntnf
      1  ... % k10
      22  ... % xm10
        99 ... % hml10
     80  ... % hntnf
      3  ... % k6
     14  ... % x66
           4  ... % k6m
     34  ... % hmpe
          31  ... % htnf6
     62  ... % xht
          21  ... % xmpe
               20  ... % xtnf6
         39  ... % sm
             101  ... % kmp
          9  ... % kma
         94  ... % kpn
     93  ... % muno
     100  ... % sM
     91  ... % kpm
     % 89  ... % kpg
];

sens_pars_25 = [ ...
     77  ... % kno
     11  ... % kmr
      7  ... % ktnf
            8  ... % ktnfm
     78  ... % xntnf
      1  ... % k10
      22  ... % xm10
        99 ... % hml10
     80  ... % hntnf
      3  ... % k6
     14  ... % x66
           4  ... % k6m
     34  ... % hmpe
          31  ... % htnf6
     62  ... % xht
          21  ... % xmpe
               20  ... % xtnf6
         39  ... % sm
             101  ... % kmp
          9  ... % kma
         94  ... % kpn
     93  ... % muno
     100  ... % sM
     91  ... % kpm
     % 89  ... % kpg
];


N = 500;

rel_bound = 0.15;


[nS,nNS,nA,labels,parS] = virtual_patients_2(sens_pars_50, N, rel_bound);
disp([nS nNS nA])


