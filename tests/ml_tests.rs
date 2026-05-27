use omega_lang::stdlib::ml::linear_regression::LinearRegression;
use omega_lang::stdlib::ml::logistic_regression::LogisticRegression;
use omega_lang::stdlib::ml::knn::KNN;
use omega_lang::stdlib::ml::decision_tree::DecisionTree;
use omega_lang::stdlib::ml::kmeans::KMeans;
use omega_lang::stdlib::ml::naive_bayes::NaiveBayes;
use omega_lang::stdlib::ml::neural_network::{NeuralNetwork, Activation};
use omega_lang::stdlib::ml::pca::PCA;

#[test]
fn test_linear_regression_simple() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut model = LinearRegression::new()
        .learning_rate(0.01)
        .epochs(5000);
    model.fit(&x, &y);

    let pred = model.predict(&[vec![6.0]]);
    assert!((pred[0] - 12.0).abs() < 1.0);
}

#[test]
fn test_linear_regression_r_squared() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0];

    let mut model = LinearRegression::new().epochs(5000);
    model.fit(&x, &y);

    let r2 = model.r_squared(&x, &y);
    assert!(r2 > 0.9);
}

#[test]
fn test_logistic_regression_and() {
    let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let y = vec![0.0, 0.0, 0.0, 1.0];

    let mut model = LogisticRegression::new()
        .learning_rate(0.5)
        .epochs(5000);
    model.fit(&x, &y);

    let predictions = model.predict(&x);
    assert_eq!(predictions[3], 1.0);
    assert_eq!(predictions[0], 0.0);
}

#[test]
fn test_knn_basic() {
    let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let y = vec![0.0, 0.0, 1.0, 1.0];

    let mut knn = KNN::new(3);
    knn.fit(&x, &y);

    let pred = knn.predict(&[vec![0.1, 0.1]]);
    assert_eq!(pred[0], 0.0);
}

#[test]
fn test_decision_tree_xor() {
    let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let y = vec![0.0, 1.0, 1.0, 0.0];

    let mut tree = DecisionTree::new().max_depth(5);
    tree.fit(&x, &y);

    let acc = tree.accuracy(&x, &y);
    assert_eq!(acc, 1.0);
}

#[test]
fn test_kmeans_two_clusters() {
    let x = vec![
        vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0],
        vec![5.0, 5.0], vec![5.1, 5.1], vec![5.0, 5.2],
    ];

    let mut kmeans = KMeans::new(2);
    kmeans.fit(&x);

    assert_ne!(kmeans.labels()[0], kmeans.labels()[5]);
}

#[test]
fn test_naive_bayes() {
    let x = vec![
        vec![1.0, 1.0], vec![1.0, 2.0], vec![2.0, 1.0],
        vec![5.0, 5.0], vec![5.0, 6.0], vec![6.0, 5.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut nb = NaiveBayes::new();
    nb.fit(&x, &y);

    let predictions = nb.predict(&[vec![1.5, 1.5], vec![5.5, 5.5]]);
    assert_eq!(predictions[0], 0);
    assert_eq!(predictions[1], 1);
}

#[test]
fn test_neural_network_xor() {
    let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let y = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut nn = NeuralNetwork::new(&[2, 8, 1])
        .learning_rate(0.5)
        .epochs(5000)
        .activation(Activation::Tanh)
        .output_activation(Activation::Sigmoid);
    nn.fit(&x, &y);

    let predictions = nn.predict(&x);
    assert!(predictions[0][0] < 0.3);
    assert!(predictions[3][0] < 0.3);
}

#[test]
fn test_pca_dimensionality_reduction() {
    let x = vec![
        vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0], vec![4.0, 5.0, 6.0],
    ];

    let mut pca = PCA::new(2);
    let transformed = pca.fit_transform(&x);

    assert_eq!(transformed[0].len(), 2);
    assert!(pca.explained_variance_ratio()[0] > 0.9);
}

#[test]
fn test_pca_reconstruction() {
    let x = vec![
        vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0], vec![4.0, 5.0, 6.0],
    ];

    let mut pca = PCA::new(3);
    let transformed = pca.fit_transform(&x);
    let reconstructed = pca.inverse_transform(&transformed);

    assert_eq!(reconstructed.len(), 4);
}
