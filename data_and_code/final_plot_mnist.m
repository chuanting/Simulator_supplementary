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
plot(train_loss, 'ko-.', 'linewidth', linewidth, 'markerfacecolor', [0, 0, 0]/255, 'markersize', markersize);
hold on;
plot(mnistflguoqing.train_loss, 'bs-.', 'linewidth', linewidth,'markerfacecolor', [0, 0, 255]/255, 'markersize', markersize)
plot(fed_with_wire_loss_mean, 'r^-.', 'linewidth', linewidth, 'markerfacecolor', [255, 0, 0]/255, 'markersize', markersize)

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



