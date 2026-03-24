%load_pars_Init_Copeland_Edited.m

function [pars, Init] = load_pars_Init_Copeland_Edited(data)
global ODE_TOL DIFF_INC 

ODE_TOL  = 1e-8; 
DIFF_INC = sqrt(ODE_TOL);



% Pathogens Models dpe
% septic: 
kpg=1.15; %4, #89| 3.882 | .6 | 1.50
peinf=20; 

kpm=1.8; %1e-1 #91|.6|.31
muno=.02; %1e-3 #93|.002|.02
sM=.012; %.005
kmp=.008; %.01
% Super sensitive
kpn=.5*.008; %1e-2 #94|1e-2 | .5*.008
% Saturation Constants %92 %Sensitive value => code to break
xI10=20; 
% Exponent %99
hmI10=2; % 3-6 similiar graphs


% Damage Model dD
kdn= 2; %.35  %95 | 2
mud= 2 ; %96 % 0.017|0.17|.37 
xdn= 10; %97 .06 | 10
hmD= 6; %107 2 
alpha= 100; %98 3|100


%%------------------Nitric oxide equation parameters------------------------
knom  = 0.002;   %  0.002|
kno   = 0.045;   % .045 |
xntnf = 100;      % 70|
hntnf = 2;       % 2|
xn10  = 5;       % 4|
hn10  = .4;     % .4 |

% NO Added PARS
knod=.002; %108 2| 0.002
xnDl10=10; %109, 30|7  Up Damage
hnDl10=1.6; %110  2|     Up Damge
xn10D=10; %111 30|      Down Il10
hn10D=2; %112        Down Il10



%% MA pars Added for Damage
kD=20*1.5;
xDam=30;
hmDa=2;
xm10D=30;
hm10D=2;






%---------------------Inflammatory Model Parameters------------------------

% TNF EQN PARA
stnf  = data.TNF(1); % Value 1 | dtnf

ktnfm = 1.5;    % 0.69712 | 2

ktnf  = .7;     % 1.4873|1

htnf6  = 6;         % dtnf|2|
xtnf6  = 560;       % dtnf|560|

htnf10 = 3;         % dtnf|3|
xtnf10 = 17.3860;   % dtnf|17.3860|

% TNF ADDED ktnfhr rate
ktnfhr= .05;


% Il-06 EQN PARA
s6    = data.IL6(1); % Value 1.53

k6m   = 3;    % 0.98831|3

k6tnf  = 2;      % 0.81|2

h6tnf  = 2;         % 2|
x6tnf  = 30;       % _|30|

h66    = 2;         % 2|
x66    = 1000;       % 560|1000

h610   = 4;         % 4|
x610   = 34.7720;   % 34.7720|

k6    = .15;    % 0.79658|.15

% Il-08 EQN PARA
k8m   = 0.07;       % Nom Val 0.560    

k8tnf  = 12;      % 0.56|8
x8tnf  = 153;       % 186|200
h8tnf  = 4;         % 4|3

x810   = 10;   % 17.3860|
h810   = 2;       % 1.5|

k8    = 3;    % 0.69839|
s8    = data.IL8(1); % Value 2.02703

% TEMP EQN PARA
tau1  = 1.6932; % 1.6932|
kt    = 0.2;    % .5|
TM    = 39.5;   % 

kttnf = 7;    % .5|
xttnf = 185;    % 185|
httnf = 2;   % 0.75|

kt6   = 3; % 1.8821|
xt6   = 560;    % 560|
ht6   = 2;   % 0.75|

kt10  = 0.0625;  % 0.0625|
xt10  = 34.7720; % 34.7720|
ht10  = 2;  % 1|    

% HR EQN PARA
HM   = data.HM;  % Value 186.7
kh   =.55;  % 0.23586|
tau2 = 2;      % 0.2|

xht  = 36.4;     % 36.4 par-62
hht  = 2;        % 2|

xhp  = 163;      % 119
hhp  = 3;        % 4


%Rate Constants
k10   = 1.1107;     % Nom Val 0.800	
	     
kma   = 2.51;	    % Not estimated
kmpe  = 4.14E-06;   % Not estimated	
kmr   = .06;      % 0.006
kpe   = 1.0819;     % Nom Val 1.01   

k10m  = 0.020536;   % Nom Val 0.019	

%Saturation Constants
xm10   = 4.3465;    % Not estimated
x106   = 560;       % Not estimated
xmtnf  = 100;       % Not estimated       
xmpe   = 3.3;       % Not estimated

%Exponents
hm10   = 0.3;	    % Not estimated 
hmtnf  = 3.160;	    % Not estimated 
hmpe   = 1;         % 1| 
h106   = 3.675;     % Not estimated  

k106   = 0.0191;    % Not estimated
kmtnf  = 4.14e-06;  % Not estimated





% Source Terms & MRmax
s10   = 0.2480;      % Not estimated
sm    = 0.0414;      % Not estimated
mmax  = 30000;       % Not estimated

%----------------------------Initial Conditions----------------------------
tnfI  = data.TNF(1); % 1
il10I = 0.2480; 
il8I  = data.IL8(1);  % 2.027
il6I  = data.IL6(1);
maI   = 0.0;   % Not estimated
mrI   = 28200; % Not estimated
peI   = 1.25;   % 1.25
DI=0;
tempI = data.temp(1); % 36.5829

Init = [tnfI il10I il8I il6I maI mrI peI ];

%-----------------------Temperature equation parameters--------------------
Tm    = tempI;  % Value 36.583


%-----------------------Temperature Initial Condition--------------------
Init = [Init tempI];

%------------------------Pain Perception Threshold pars--------------------
kpepp =  0.16;  % Nom Val 0.2
kpp   =  0.15;  % Nom Val 0.075 
ppM   = 713.56; % Not estimated

Init = [Init ppM];

%---------------------Cardiovascular Model Parameters----------------------
% Height, weight and gender used to predict blood volume
H      = data.height;  % Subject height cm
W      = data.weight;  % Subject weight kg
Gender = 1;   % Male 1, female 2

% Define body surface area, used to predict blood volume
BSA = sqrt(H*W/3600);  % Body surface area

% Prediction of patient specific blood volume and cardiac output
if Gender == 2
  TotalVol = (3.47*BSA - 1.954)*1000; % Female ml/min
elseif Gender == 1
  TotalVol = (3.29*BSA - 1.229)*1000; % Male ml/min
end;
qtot = TotalVol/60;     % Assume blood circulates in one min ml/sec
CO   = TotalVol/1000;   % l/sec

% Flows (related to subject)
qa  = qtot;        % Arteries --> arteries in organ bed
qs  = qtot;        % Arteries in organ bed --> veins in organ bed
qv  = qtot;        % Veins in organ bed --> veins

% Pressures (related to subject) 
pla = data.BP(1)-5.9840;% Organ bed systolic arterial pressure
psa = pla*.95;   % Arterial systolic pressure (ideally should come from data!)
psv = 3.75;      % Organ bed veins
plv = 3.5;       % Venous pressure

% Resistances (Ohm's law)
Ra  = 0.065755;  % Nom Val (pla - psa)/qa;   % Arteries --> organ bed arteries
Rv  = 0.0028345; % Nom Val (psv - plv)/qv;   % Organ bed veins --> veins
Rs  = 1.0789;    % Nom Val (psa - psv)/qs;   % Arteries in organ bed --> veins in organ bed

% Volume distribution (Beneken and deWit)
VtotS = TotalVol;  % Systemic volume is 85% of total volume 
VtotA = 0.20*VtotS;     % Arterial volume is 20% of systemic volume
VtotV = 0.80*VtotS;     % Venous volume is 80% of systemic volume

VlaT  = VtotA*0.85;
VsaT  = VtotA*0.15;
VlvT  = VtotV*0.15;
VsvT  = VtotV*0.85;

VlaS   = VlaT*0.18;
VsaS   = VsaT*0.18;
VlvS   = VlvT*0.05;
VsvS   = VsvT*0.05;
VStot  = VlaS + VlvS + VsaS + VsvS;

Vun    = 10;

% Compliances, stressed volume percentages from Beneken are weighted averages (Beneken and deWit)
Cla = 1.3961;  % Nom Val VlaS/pla;  % Artery compliance (stressed to total volume arteries ~25%)
Clv = 9.0718;  % Nom Val VlvS/plv;  % Venous compliance (stressed to total volume veins ~7%)
Csa = 0.25934; % Nom Val VsaS/psa;  % Organ bed artery compliance 
Csv = 47.9795; % Nom Val VsvS/psv;  % Organ bed venous compliance

% Parameters needed to set up information within the heart (Olufsen and Williams non-puls)
Vun  = 10;		    % Unstressed ventricular volume
Ved  = 142 - Vun;  	% Maximum ventricular volume (volume at end of filling), can we make this patient specific?
Ves  = 47  - Vun;
Em   = 0.026515;    % Nom Val plv/Ved;     % Minimum elasticity of the heart [ pv / (Ved - Vun) ]
EM   = 3.13481;     % Nom Val pla/Ves;

HI = data.hr(1);    % Value 64.28570
hr = HI/60;

VT   = VStot;
VlaI = (Ra*hr + Rs*hr + Rv*hr + Em)*Cla*EM*VT/(Cla*EM*Ra*hr + Cla*EM*Rs*hr + Cla*EM*Rv*hr + Clv*Em*Ra*hr + Clv*Em*Rs*hr + Clv*Em*Rv*hr + Csa*EM*Rs*hr + Csa*EM*Rv*hr + Csa*Em*Ra*hr + Csv*EM*Rv*hr + Csv*Em*Ra*hr + Csv*Em*Rs*hr + Cla*EM*Em + Clv*EM*Em + Csa*EM*Em + Csv*EM*Em);
VlvI = Clv*Em*VT*(Ra*hr + Rs*hr + Rv*hr + EM)/(Cla*EM*Ra*hr + Cla*EM*Rs*hr + Cla*EM*Rv*hr + Clv*Em*Ra*hr + Clv*Em*Rs*hr + Clv*Em*Rv*hr + Csa*EM*Rs*hr + Csa*EM*Rv*hr + Csa*Em*Ra*hr + Csv*EM*Rv*hr + Csv*Em*Ra*hr + Csv*Em*Rs*hr + Cla*EM*Em + Clv*EM*Em + Csa*EM*Em + Csv*EM*Em);
VsvI = Csv*VT*(EM*Rv*hr + Em*Ra*hr + Em*Rs*hr + EM*Em)/(Cla*EM*Ra*hr + Cla*EM*Rs*hr + Cla*EM*Rv*hr + Clv*Em*Ra*hr + Clv*Em*Rs*hr + Clv*Em*Rv*hr + Csa*EM*Rs*hr + Csa*EM*Rv*hr + Csa*Em*Ra*hr + Csv*EM*Rv*hr + Csv*Em*Ra*hr + Csv*Em*Rs*hr + Cla*EM*Em + Clv*EM*Em + Csa*EM*Em + Csv*EM*Em);
VsaI = VT-VlaI-VsvI-VlvI;


% Start the system in steady state
Init = [Init VlaI VsaI VlvI VsvI];

%------------------------Heart rate equation parameters--------------------


Init = [Init HI];


noI   = 0;
Init = [Init noI];

%-------------------Resistance equation parameters-------------------------
krpp = 19.7806;  % Nom Val 10 
xrpp = 120;      % Not estimated (Personalized) 
hrpp = 2;        % Not estimated
krno = 2;        % Not estimated (Personalized)
kr   = 7.15076;  % Nom Val 2 
% BPI=0;

Init = [Init Rs DI];

%Parameter vector
pars = [k10   k10m   k6    k6m   k8      k8m    ktnf  ktnfm kma    kmpe ...  %1 - 10
        kmr   kpe    x610  x66   x6tnf   x810   x8tnf x106  xtnf10 xtnf6 ... %11-20
        xmpe  xm10   xmtnf h106  h6tnf   h66    h610  h8tnf h810   htnf10 ...%21-30
        htnf6 hm10   hmtnf hmpe  stnf    s10    s8    s6    sm     mmax ...  %31-40
        k6tnf k8tnf  k106  kmtnf                                        ...  %41-44 %end of inflammatory
        tau1  TM     Tm    kt    kttnf   kt6    kt10  xttnf xt6    xt10 ...  %45-54
        httnf ht6    ht10                                               ...; %55-57 %end of temperature
        tau2  HM     HI    kh    xht     hht                            ...';%58-63 %end of heart rate
        ppM   kpepp  kpp                                                ...';%64-66 %end of PPT 
        Ra    Rv     Rs    Cla    Csa    Clv     Csv   Em    EM         ...';%67-75 %end of BP
        knom  kno    xntnf xn10  hntnf   hn10                           ...';%76-81 %end of NO
        krpp  krno   kr    xrpp  hrpp    xhp    hhp     ...%82-88 
        kpg peinf kpm xI10 muno kpn ...%89-94
        kdn mud xdn alpha hmI10 ...        %95-99                    
        sM  kmp               ... %100-101
        kD xDam hmDa xm10D hm10D hmD knod xnDl10 hnDl10 xn10D hn10D ktnfhr                                       ]'; %102-112