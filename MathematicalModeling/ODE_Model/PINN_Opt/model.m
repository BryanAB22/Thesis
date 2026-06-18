
%model.m

% function xdot = model(t,y,Z,pars)
function xdot = model(t,y,pars)


%----------------------------------Lags------------------------------------
        
limit_idx = [1 2 3 4 5 6 7 15 17];  % tnf il10 cxcl8 il6 ma mr pe no D

y(limit_idx) = max(y(limit_idx), 0);

%---------------------------------States-----------------------------------
tnf   = y(1);  
il10  = y(2);  
cxcl8 = y(3);  
il6   = y(4);
ma    = y(5);  
mr    = y(6);  
pe    = y(7);
temp2 = y(8);
pp    = y(9);
Vla   = y(10); 
Vsa   = y(11); 
Vlv   = y(12); 
Vsv   = y(13);
hr    = y(14);
no    = y(15); 
rs    = y(16);
D     =y(17);

%-------------------------------Parameters---------------------------------
% Immune / cytokines (1–44)
k10    = pars(1);
k10m   = pars(2);
k6     = pars(3);
k6m    = pars(4);
k8     = pars(5);
k8m    = pars(6);

ktnf   = pars(7);
ktnfm  = pars(8);
kma    = pars(9);
kmpe   = pars(10);
kmr    = pars(11);
kpe    = pars(12);

x610   = pars(13);
x66    = pars(14);
x6tnf  = pars(15);
x810   = pars(16);
x8tnf  = pars(17);
x106   = pars(18);

xtnf10 = pars(19);
xtnf6  = pars(20);
xmpe   = pars(21);
xm10   = pars(22);
xmtnf  = pars(23);

h106   = pars(24);
h6tnf  = pars(25);
h66    = pars(26);
h610   = pars(27);
h8tnf  = pars(28);
h810   = pars(29);
htnf10 = pars(30);
htnf6  = pars(31);
hm10   = pars(32);
hmtnf  = pars(33);
hmpe   = pars(34);

stnf   = pars(35);
s10    = pars(36);
s8     = pars(37);
s6     = pars(38);
sm     = pars(39);
mmax   = pars(40);

k6tnf  = pars(41);
k8tnf  = pars(42);
k106   = pars(43);
kmtnf  = pars(44);

% Temperature module (45–57)
tau1   = pars(45);
TM     = pars(46);
Tm     = pars(47);
kt     = pars(48);
kttnf  = pars(49);
kt6    = pars(50);
kt10   = pars(51);
xttnf  = pars(52);
xt6    = pars(53);
xt10   = pars(54);
httnf  = pars(55);
ht6    = pars(56);
ht10   = pars(57);

% Heart-rate / baroreflex (58–63, 87–88, 60 also used)
tau2   = pars(58);
HM     = pars(59);
Hm     = pars(60);
kh     = pars(61);
xht    = pars(62);
hht    = pars(63);
xhp    = pars(87);
hhp    = pars(88);
BPo=pars(114);
% Pain perception (64–66)
ppM    = pars(64);
kpepp  = pars(65);
kpp    = pars(66);

% Cardiovascular volumes/pressures (67–75)
Ra     = pars(67);
Rv     = pars(68);
Rs     = pars(69);
Cla    = pars(70);
Csa    = pars(71);
Clv    = pars(72);
Csv    = pars(73);
Em     = pars(74);
EM     = pars(75);

% Nitric oxide + resistance feedback (76–86)
knom   = pars(76);
kno    = pars(77);
xntnf  = pars(78);
xn10   = pars(79);
hntnf  = pars(80);
hn10   = pars(81);
krpp   = pars(82);
krno   = pars(83);
kr     = pars(84);
xrpp   = pars(85);
hrpp   = pars(86);

% Pathogen / PE dynamics (89–94, 99–101, 113)
kpg    = pars(89);
peinf  = pars(90);
kpm    = pars(91);
xI10   = pars(92);
muno   = pars(93);
kpn    = pars(94);

hmI10  = pars(99);
sM     = pars(100);
kmp    = pars(101);

ktnfhr = pars(113);

% Damage dynamics (95–98, 102–107)
kdn    = pars(95);
mud    = pars(96);
xdn    = pars(97);
alpha  = pars(98);

kD     = pars(102);
xDam   = pars(103);
hmDa   = pars(104);
xm10D  = pars(105);
hm10D  = pars(106);
hmD    = pars(107);

%NO/Damage couplings (108–112)
knod   = pars(108);
xnDl10 = pars(109);
hnDl10 = pars(110);
xn10D  = pars(111);
hn10D  = pars(112);



%----------------------------State Equations------------------------------- 
dtnf   = ktnfm*ma*fs(il10,xtnf10,htnf10)*fs(il6,xtnf6,htnf6)*( 1 + ktnfhr*abs((hr - Hm))) - ktnf*(tnf - stnf);
dil6   = ma*(k6m + k6tnf*f(tnf,x6tnf,h6tnf))*fs(il6,x66,h66)*fs(il10,x610,h610) - k6*(il6 - s6);
dil8   = ma*(k8m + k8tnf*f(tnf,x8tnf,h8tnf))*fs(il10,x810,h810) - k8*(cxcl8 - s8);
dil10  = ma*(k10m + k106*f(il6,x106,h106)) - k10*(il10 - s10);
dmr    = kmr*mr*(1-(mr/mmax)) - (f(pe,xmpe,hmpe))*(sm + kmtnf*f(tnf,xmtnf,hmtnf))*fs(il10,xm10,hm10)*mr;
% dma    = (f(pe,xmpe,hmpe))*(sm + kmtnf*f(tnf,xmtnf,hmtnf)*(xmtnf)^hmtnf)*fs(il10,xm10,hm10)*mr+ kD*fs(il10,xm10D,hm10D)*f(D,xDam,hmDa) - kma*ma;   
% dma    = (f(pe,xmpe,hmpe))*(sm + kmtnf*f(tnf,xmtnf,hmtnf))*fs(il10,xm10,hm10)*mr+ kD*fs(il10,xm10D,hm10D)*f(D,xDam,hmDa) - kma*ma; 


dma    = (f(pe,xmpe,hmpe))*(sm + kmtnf*f(tnf,xmtnf,hmtnf))*fs(il10,xm10,hm10)*mr+ kD*fs(il10,xm10D,hm10D)*D - kma*ma;   


dpe    = kpg * pe * (1 - pe/peinf) ...
          - 1*(kpm*sM*pe)/(muno+kmp*pe) ...
          - kpn*ma*pe*fs(il10,xI10,hmI10);

dD= kdn*f(alpha*no+pe,xdn,hmD) - mud*D;


xdot = [dtnf;dil10; dil8; dil6; dma; dmr;dpe];

%----------------------------Temperature RHS-------------------------------
%-------------------------------Parameters---------------------------------

F      = kt*(TM-Tm)*(kttnf*f((abs(tnf-stnf)),xttnf,httnf) + kt6*f((abs(il6 - s6)),xt6,ht6) - kt10*(1-fs((abs(il10-s10)),xt10,ht10))) + Tm;

dtemp2 = ((-temp2 + F)/tau1);

xdot   = [xdot; dtemp2];

%---------------------------Pain Perception RHS----------------------------
%-------------------------------Parameters---------------------------------
dpp   = -kpepp*pe*pp + kpp*(ppM-pp);

xdot = [xdot; dpp];

%----------------------------Cardio Model  RHS-----------------------------
%-------------------------------Parameters---------------------------------

% Calculate pressures (stressed volume only!)
pla  = Vla/Cla;
psa  = Vsa/Csa;
plv  = Vlv/Clv;
psv  = Vsv/Csv;

%Calculate flows
qa = (pla - psa)/Ra;
qs = (psa - psv)/rs;
qv = (psv - plv)/Rv;

Vstr = -(pla/EM - plv/Em); 
Q    = Vstr*hr/60;


               %Vla     %Vsa    %Vlv    %Vsv
xdot = [xdot; Q - qa; qa - qs; qv - Q;  qs - qv];

%-----------------------------Heart Rate RHS-------------------------------
%-------------------------------Parameters---------------------------------
%
% Smooth differentiable replacement for the original hard switch:
%     if (Vla/Cla) > 100      use pressureHigh
%     if (Vla/Cla) <= 100     use pressureLow
%
% This keeps the same two pressure terms, but blends them smoothly near
% bp = 100. Far from 100, the result is almost the same as the original
% if/else model. This is safer for PINNs and gradient-based fitting.

bp = Vla/Cla;
epsSmooth = 1e-8;

absTemp = smoothAbs(temp2 - Tm);
absXht  = smoothAbs(xht - Tm);
tempTerm = absTemp.^hht ./ (absTemp.^hht + absXht.^hht + epsSmooth);

absBp  = smoothAbs(bp - BPo);
absXhp = smoothAbs(xhp - BPo);
pressureDenom = absBp.^hhp + absXhp.^hhp + epsSmooth;

% Original branch when bp > 100.
pressureHigh = absXhp.^hhp ./ pressureDenom;

% Original branch when bp <= 100.
pressureLow = absBp.^hhp ./ pressureDenom;

% Larger value = sharper switch. Smaller value = smoother/wider transition.
switchSharpness = 0.25;
wHigh = 1 ./ (1 + exp(-switchSharpness*(bp - 100)));

pressureTerm = wHigh.*pressureHigh + (1 - wHigh).*pressureLow;
ft = kh*(HM - Hm).*tempTerm.*pressureTerm + Hm;

xdot = [xdot; (-hr + ft)/tau2];


%----------------------------Nitric Oxide RHS------------------------------
%-------------------------------Parameters---------------------------------

% dno   = knom*ma*f(tnf,xntnf,hntnf)*fs(il10,xn10,hn10)+knod*f(D,xnDl10,hnDl10)*fs(il10,xn10D,hn10D) - kno*no;
dno   = knom*ma*f(tnf,xntnf,hntnf)*fs(il10,xn10,hn10)+knod*D*fs(il10,xn10D,hn10D) - kno*no;


xdot = [xdot; dno];

%-----------------------------Resistance RHS-------------------------------
%-------------------------------Parameters---------------------------------

drs   = krpp*(dpp^hrpp/(dpp^hrpp + xrpp^hrpp)) - krno*no - kr*(rs - Rs);


xdot = [xdot; drs; dD];

for k = limit_idx
    if y(k) < 0
        y(k)=0;
        xdot(k) = 0;
    end
end



function [y] = fs(v,x,hill) 
  y = x.^hill./(v.^hill + x.^hill); 


function [y] = f(v,x,hill)
  y  = (v).^hill./(v.^hill + x.^hill);


function [y] = smoothAbs(x)
  y = sqrt(x.^2 + 1e-8);

