clear;
close all;
clc;

markersize = 4;
linewidth = 1;
fontsize = 9;

sys_batch_size = readtable('./data/sys_performance_batch_size.csv');
threshold_1 = sys_batch_size(sys_batch_size.threshold==0.2, :);
threshold_2 = sys_batch_size(sys_batch_size.threshold==0.15, :);
threshold_3 = sys_batch_size(sys_batch_size.threshold==0.10, :);

x = [10 20 40 80 160];
subplot(2, 2, 1)
semilogx(x, threshold_1{:, 3}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 0]/255, 'markersize', markersize)
hold on;
semilogx(x, threshold_2{:, 3}, 'bs-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
semilogx(x, threshold_3{:, 3}, 'r^-.', 'linewidth', linewidth, 'markerfacecolor', [255, 0, 0]/255, 'markersize', markersize)
xticks(x)
grid on;
legend('loss < 0.2', 'loss < 0.15', 'loss < 0.1')
xlabel('Number of clients')
ylabel('Throughput (Mb/s)')


subplot(2, 2, 2)
semilogx(x, threshold_1{:, 4}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 0]/255, 'markersize', markersize)
hold on;
semilogx(x, threshold_2{:, 4}, 'bs-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
semilogx(x, threshold_3{:, 4}, 'r^-.', 'linewidth', linewidth, 'markerfacecolor', [255, 0, 0]/255, 'markersize', markersize)
xticks(x)
grid on;
legend('loss < 0.2', 'loss < 0.15', 'loss < 0.1')
xlabel('Number of clients')
ylabel('Energy (J)')


subplot(2, 2, 3)
semilogx(x, threshold_1{:, 5}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 0]/255, 'markersize', markersize)
hold on;
semilogx(x, threshold_2{:, 5}, 'bs-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
semilogx(x, threshold_3{:, 5}, 'r^-.', 'linewidth', linewidth, 'markerfacecolor', [255, 0, 0]/255, 'markersize', markersize)
xticks(x)
grid on;
legend('loss < 0.2', 'loss < 0.15', 'loss < 0.1')
xlabel('Number of clients')
ylabel('Time (s)')



subplot(2, 2, 4)
semilogx(x, threshold_1{:, 6}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 0]/255, 'markersize', markersize)
hold on;
semilogx(x, threshold_2{:, 6}, 'bs-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
semilogx(x, threshold_3{:, 6}, 'r^-.', 'linewidth', linewidth, 'markerfacecolor', [255, 0, 0]/255, 'markersize', markersize)
xticks(x)
grid on;
legend('loss < 0.2', 'loss < 0.15', 'loss < 0.1')
xlabel('Number of clients')
ylabel('Packet loss ratio')
% pos=get(gca,'position');  % retrieve the current values
% pos(2)=0.9*pos(2);        % try reducing width 10%
% set(gca,'position',pos);  % write the new values

