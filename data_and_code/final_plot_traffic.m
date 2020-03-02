clear;
close all;
clc;

%% Prediction accuracy comparisons
federated_with_wireless = readtable('./data/traffic_exp_epsilon=0.1_clients=10_channel=wireless_stop=False_condition=0.5_acc.csv');
markersize = 4;
linewidth = 1;
fontsize = 9;
load ./data/traffic_centralized_guoqing.mat
load ./data/traffic_fl_guoqing.mat

fed_loss = readtable('./data/traffic_exp_epsilon=0.1_clients=10_channel=wireless_stop=False_condition=0.5_train_loss.csv');
fed_with_wireless_loss = mean(fed_loss{2:end, :}, 2);

centralized_loss = trafficcentralizedguoqing.train_loss;
fed_no_wireless_loss = trafficflguoqing.train_loss;

plot(centralized_loss, 'ko-.', 'linewidth', linewidth, 'markersize', markersize);
hold on;
plot(fed_no_wireless_loss, 'bs-.', 'linewidth', linewidth, 'markersize', markersize)
plot(fed_with_wireless_loss, 'r^-.', 'linewidth', linewidth, 'markersize', markersize)

grid on;
legend('Centralized', 'Federated (No wireless)', 'Federated (With wireless)')
xlabel('Training epochs')
ylabel('Average training loss')
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');

%% QoS
figure
subplot(321)
plot(federated_with_wireless{:, 4}, federated_with_wireless{:, 6}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
% set(gca,'YDir','reverse')
set(gca, 'XDir', 'reverse')
grid on;
xlabel('MSE')
ylabel('Energy (J)')

subplot(322)
plot(federated_with_wireless{:, 4}, federated_with_wireless{:, 7}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
set(gca, 'XDir', 'reverse')
grid on;
xlabel('MSE')
ylabel('Time (s)')

subplot(323)
bar(federated_with_wireless{:, 8})
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Training epochs')
ylabel('Throughput (Mb/s)')
subplot(324)
plot(federated_with_wireless{:, 9}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');
grid on;
xlabel('Training epochs')
ylabel('Packet loss ratio')
xlim([0 51])
subplot(3, 2, [5, 6])
% bar([federated_with_wireless{:, 3} 10-federated_with_wireless{:, 3}], 0.8, 'stack');
plot(10-federated_with_wireless{:, 3}, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize);
% legend('Uneffective links', 'Effective links')
xlabel('Training epochs')
ylabel('Number of links')
grid on;
set(gca, 'linewidth', linewidth, 'fontsize', fontsize, 'fontname', 'Arial');

