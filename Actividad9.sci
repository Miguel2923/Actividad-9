// Cargar datos desde el archivo
data = read("reprocessed.hungarian.data", -1, -1);

// Reemplazar valores faltantes (-9) por la media de la columna
for i = 1:size(data, 2)
    idx = find(data(:, i) == -9);
    if ~isempty(idx) then
        col_mean = mean(data(data(:, i) <> -9, i));
        data(idx, i) = col_mean;
    end
end

// Normalización de los datos (Min-Max Scaling)
for i = 1:size(data, 2) - 1
    min_val = min(data(:, i));
    max_val = max(data(:, i));
    data(:, i) = (data(:, i) - min_val) / (max_val - min_val);
end

// Separar características (X) y etiquetas (Y)
X = data(:, 1:13);
Y = data(:, 14);

// Convertir etiquetas a formato one-hot
num_classes = max(Y) + 1;
Y_onehot = zeros(size(Y, 1), num_classes);
for i = 1:size(Y, 1)
    Y_onehot(i, Y(i) + 1) = 1;
end

// Crear la red neuronal
layers = [13, 10, 5, num_classes]; // 13 entradas, 2 capas ocultas (10 y 5 neuronas), salida con num_classes
net = ann_new(layers, "Logistic");

// Entrenar la red neuronal
eta = 0.01; // Tasa de aprendizaje
epochs = 1000;
[net, err] = ann_train_Bprop(net, X, Y_onehot, eta, epochs);

// Evaluar la red neuronal
Y_pred = ann_sim(net, X);
[~, predictions] = max(Y_pred, [], 2);
predictions = predictions - 1;

// Calcular precisión
accuracy = sum(predictions == Y) / length(Y);
printf("Precisión del modelo: %.2f%%\n", accuracy * 100);
