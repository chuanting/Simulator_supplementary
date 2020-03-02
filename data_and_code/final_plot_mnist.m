clear;
close all;
clc;

%% Prediction accuracy comparisons
federated_with_wireless = readtable('./data/mnist_exp_epsilon=0.1_clients=10_channel=wireless_seed=20_acc.csv');

markersize = 4;
linewidth = 1;
fontsize = 9;

%% Traning loss comparisons
load ./data/mnist_fl_guoqing.mat;
fed_with_wire_loss = readtable('./data/mnist_exp_epsilon=0.1_clients=10_channel=wireless_seed=20_train_loss.csv');
fed_with_wire_loss_mean = mean(fed_with_wire_loss{2:end, :}, 2);
figure

load ./data/mnist_centralized_guoqing.mat;
plot(train_loss, 'ko-.', 'linewidth', linewidth, 'markersize', markersize);
hold on;
plot(mnistflguoqing.train_loss, 'bs-.', 'linewidth', linewidth, 'markersize', markersize)
plot(fed_with_wire_loss_mean, 'r^-.', 'linewidth', linewidth, 'markersize', markersize)

grid on;
legend('Centralized', 'Federated (No wireless)', 'Federated (With wireless)')
xlabel('Traning epochs')
ylabel('Average training loss')
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');

%% QoS
figure
subplot(321)
plot(federated_with_wireless{:, 4},federated_with_wireless{:, 5}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Accuracy')
ylabel('Energy (J)')

subplot(322)
plot(federated_with_wireless{:, 4},federated_with_wireless{:, 6}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Accuracy')
ylabel('Time (s)')

subplot(323)
bar(federated_with_wireless{:, 7})
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Training epochs')
ylabel('Throughput (Mb/s)')
subplot(324)
plot(federated_with_wireless{:, 8}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Training epochs')
ylabel('Packet loss ratio')
xlim([0 51])
subplot(3, 2, [5, 6])
% bar([federated_with_wireless{:, 3} 10-federated_with_wireless{:, 3}], 0.8, 'stack');
plot(10-federated_with_wireless{:, 3}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize);
% legend('Effective links')
xlabel('Training epochs')
ylabel('Number of links')
grid on;
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');

%% clients
% figure
% 
% mnist_goodput_1 = [89.98574693284421, 67.49369315065024, 91.29795106420431, 99.37720385661352, 96.55301458747421];
% energy_cost_1 = [3.51758320894638, 7.825609756002059, 12.327876818154364, 31.815321738259286, 131.08945739148734];
% time_cost_1 = [468.70390624999925, 506.2002187499992, 393.7112812499994, 487.4520624999992, 941.1574437499984];
% packet_loss_1 = [0.31666666666666665, 0.35862068965517246, 0.34293478260869564, 0.3441964285714286, 0.3595486111111111];
% 
% mnist_goodput_2 = [86.34669304935612, 75.38485928949962, 87.4139279201562, 94.63934906978014, 97.21074364534923];
% energy_cost_2 = [3.9507321893762035, 10.751719632764903, 20.150142629240552, 63.97639599834212, 220.61721021114033];
% time_cost_2 = [524.9483749999991, 701.1810437499988, 644.9365749999989, 978.6537562499983, 1582.344387500031];
% packet_loss_2 = [0.3216666666666666, 0.35125000000000006, 0.34966216216216206, 0.34787946428571426, 0.3582291666666667];
% 
% mnist_goodput_3 = [87.27164806794067, 79.52729353491505, 90.6400358439412, 96.72771122435816, 97.08030277984001];
% energy_cost_3 = [9.881977633691145, 18.949015797126126, 46.55858907875532, 163.89470547703763, 653.867268962821];
% time_cost_3 = [1312.370937500015, 1237.3783125000107, 1488.6036062500255, 2508.5033062500297, 4679.539800000036];
% packet_loss_3 = [0.32733333333333353, 0.3464788732394367, 0.34367647058823525, 0.34567307692307686, 0.35896381578947356];
% 
% 
% plot(mnist_goodput_1, 'ro-')
% hold on;
% plot(mnist_goodput_2, 'ko-')
% plot(mnist_goodput_3, 'bo-');


% 
% b1 = bar([mnist_goodput_1; mnist_goodput_1; mnist_goodput_3]);
% grid on;
% ylim([65 inf])
% legend('10 clients', '20 clients', '40 clients', '80 clients', '160 clients')
% xticklabels({'0.2', '0.15', '0.1'})
% xlabel('Loss threshold')
% ylabel('System goodput')
% 
% figure
% 
% b2 = bar([energy_cost_1; energy_cost_2; energy_cost_3]);
% grid on;
% ylim([0 700])
% legend('10 clients', '20 clients', '40 clients', '80 clients', '160 clients')
% xticklabels({'0.2', '0.15', '0.1'})
% xlabel('Loss threshold')
% ylabel('Total power consumed')
% 
% figure
% b3 = bar([time_cost_1; time_cost_2; time_cost_3]);
% grid on;
% legend('10 clients', '20 clients', '40 clients', '80 clients', '160 clients')
% xticklabels({'0.2', '0.15', '0.1'})
% xlabel('Loss threshold')
% ylabel('Total time')
% 
% 
% figure
% b4 = bar([packet_loss_1; packet_loss_2; packet_loss_3]);
% grid on;
% ylim([0.2 inf])
% legend('10 clients', '20 clients', '40 clients', '80 clients', '160 clients')
% xticklabels({'0.2', '0.15', '0.1'})
% xlabel('Loss threshold')
% ylabel('Packet loss')




