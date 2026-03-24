clear; clc; close all;

global Init Tm BPo

load DataCopeland.mat

BPtime = BPt;  BP = BPm;  BPsd = BPse*sqrt(10);  data.BP = BP; %#ok<NASGU>
HRtime = HRt;  HR = HRm;  HRsd = HRse*sqrt(10);  data.hr = HR; %#ok<NASGU>
IMMUNEtime = TNFt; %#ok<NASGU>
TNF = TNFm;   TNFsd = TNFse*sqrt(10); data.TNF = TNF; %#ok<NASGU>
IL6 = IL6m;   IL6sd = IL6se*sqrt(10); data.IL6 = IL6; %#ok<NASGU>
IL8 = IL8m;   IL8sd = IL8se*sqrt(10); data.IL8 = IL8; %#ok<NASGU>
TEMPtime = TEMPt(1:7); %#ok<NASGU>
TEMP     = TEMPm(1:7);
TEMPsd   = TEMPse(1:7) * sqrt(10); %#ok<NASGU>

data.temp   = TEMP;
data.age    = 29;
data.weight = 79.9;
data.height = 177;
data.HM     = 207 - 0.7 * data.age;

BPo = BP(1);
Tm  = data.temp(1);

[pars, Init] = load_pars_Init_Copeland_Edited(data);

pars0 = pars;
Init0 = Init;

time = [0 5000];

kpg_idx = 89;

Ma_idx     = 5;
Pe_idx     = 7;
Damage_idx = 17;

threshold = 3e-3;

P0_grid  = linspace(0, 2.5, 50);
kpg_grid = linspace(0, 2.5, 50);

label = zeros(numel(P0_grid), numel(kpg_grid));

tic
%% kpg vs Pe_0 
for ik = 1:numel(P0_grid)
    Init = Init0;
    Init(Pe_idx) = P0_grid(ik);

    for ip = 1:numel(kpg_grid)
        pars = pars0;
        pars(kpg_idx) = kpg_grid(ip);

        try
            [t, y] = modelDriver(pars, Init, time);

            if isempty(t) || t(end) < time(end) - 1e-9
                label(ik,ip) = -2;
                continue
            end

            if any(isnan(y(:))) || any(isinf(y(:)))
                label(ik,ip) = -3;
                continue
            end

            Ma     = y(end, Ma_idx);
            Pe     = y(end, Pe_idx);
            Damage = y(end, Damage_idx);

            if (Ma <= threshold) && (Pe <= threshold) && (Damage <= threshold)
                label(ik,ip) = 1;      
            elseif (Pe <= threshold) && (Ma > threshold) && (Damage > threshold)
                label(ik,ip) = 2;
            elseif (Pe > threshold) && (Ma > threshold) && (Damage > threshold)
                label(ik,ip) = 3;      
            else
                label(ik,ip) = 4;      
            end

        catch
            label(ik,ip) = -1;         
        end
    end
end

toc
figure;
imagesc(kpg_grid, P0_grid, label);
set(gca,'YDir','normal');
xlabel('K_{pg}');
ylabel('Pe_{0}');
title('K_{pg} vs Pe_{0}');
colorbar; grid on;
% warning off
