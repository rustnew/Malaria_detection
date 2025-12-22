use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[usize; 2]; 2],
}

impl ClassificationMetrics {
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            confusion_matrix: [[0, 0], [0, 0]],
        }
    }
    
    pub fn calculate(&mut self, predictions: &[usize], targets: &[usize]) {
        let mut tp = 0; // True Positives (Parasitized correctly classified)
        let mut tn = 0; // True Negatives (Uninfected correctly classified)
        let mut fp = 0; // False Positives (Uninfected classified as Parasitized)
        let mut fn_val = 0; // False Negatives (Parasitized classified as Uninfected)
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            match (*pred, *target) {
                (0, 0) => tp += 1,  // Correctly identified as Parasitized
                (1, 1) => tn += 1,  // Correctly identified as Uninfected
                (0, 1) => fp += 1,  // Uninfected classified as Parasitized
                (1, 0) => fn_val += 1, // Parasitized classified as Uninfected
                _ => {}
            }
        }
        
        self.confusion_matrix = [[tp, fn_val], [fp, tn]];
        
        let total = predictions.len() as f64;
        self.accuracy = (tp + tn) as f64 / total;
        
        self.precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        
        self.recall = if tp + fn_val > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        
        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        } else {
            0.0
        };
    }
    
    pub fn print_summary(&self) {
        println!("=== Classification Metrics ===");
        println!("Accuracy:  {:.4}", self.accuracy);
        println!("Precision: {:.4}", self.precision);
        println!("Recall:    {:.4}", self.recall);
        println!("F1-Score:  {:.4}", self.f1_score);
        println!("Confusion Matrix:");
        println!("                Predicted");
        println!("                P     U");
        println!("Actual  P     {:4}  {:4}", self.confusion_matrix[0][0], self.confusion_matrix[0][1]);
        println!("        U     {:4}  {:4}", self.confusion_matrix[1][0], self.confusion_matrix[1][1]);
        println!("==============================");
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub learning_rate: f64,
}

pub struct MetricsTracker {
    history: VecDeque<TrainingMetrics>,
    max_history: usize,
}

impl MetricsTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }
    
    pub fn add(&mut self, metrics: TrainingMetrics) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }
    
    pub fn get_best_epoch(&self) -> Option<&TrainingMetrics> {
        self.history.iter().max_by(|a, b| {
            a.val_accuracy.partial_cmp(&b.val_accuracy).unwrap()
        })
    }
    
    pub fn print_history(&self) {
        println!("┌─────────┬────────────┬────────────┬──────────────┬──────────────┬────────────┐");
        println!("│ Epoch   │ Train Loss │ Val Loss   │ Train Acc    │ Val Acc      │ LR         │");
        println!("├─────────┼────────────┼────────────┼──────────────┼──────────────┼────────────┤");
        
        for metrics in &self.history {
            println!("│ {:7} │ {:10.4} │ {:10.4} │ {:12.4} │ {:12.4} │ {:10.6} │",
                metrics.epoch,
                metrics.train_loss,
                metrics.val_loss,
                metrics.train_accuracy,
                metrics.val_accuracy,
                metrics.learning_rate
            );
        }
        println!("└─────────┴────────────┴────────────┴──────────────┴──────────────┴────────────┘");
        
        if let Some(best) = self.get_best_epoch() {
            println!("Best epoch: {} with validation accuracy: {:.4}%", 
                best.epoch, best.val_accuracy * 100.0);
        }
    }
}