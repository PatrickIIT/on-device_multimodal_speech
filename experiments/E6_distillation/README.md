# Experiment E6: Cross-Architecture Knowledge Distillation

## Objective
Train a compact student streaming acoustic model under the supervision of a high-capacity teacher model using soft target distillation and intermediate representation matching.

## Hypothesis
Knowledge distillation will transfer acoustic representations from the teacher to a 50% smaller student model, recovering over 60% of the WER gap between the small student baseline and the large teacher model.

## Variables
* **Independent Variable**: Distillation loss formulation (Hard targets vs. Soft logits vs. Multi-loss with intermediate feature matching).
* **Dependent Variables**: Student WER, CER, parameter efficiency, convergence rate.
* **Control Variables**: Student architecture (compact 8-layer Conformer), dataset, training optimizer.

## Dataset
* Mozilla Common Voice Swahili (Train/Dev/Test).

## Model
* **Teacher**: Large Conformer / Whisper-medium (frozen).
* **Student**: Compact streaming Conformer (8 layers, 144 dim).

## Procedure
1. Pre-train or freeze the high-capacity teacher model.
2. Configure distillation loss weights ($\alpha_{CE} = 0.4, \alpha_{CTC} = 0.3, \alpha_{KD} = 0.3, T=2.0$).
3. Train student model for 30 epochs on Common Voice Swahili.
4. Evaluate acoustic recognition accuracy on test splits.

## Metrics
* WER, CER, Parameter Count, Training Convergence Curves, RTF.

## Expected Output Files
* `results/raw/E6_distillation_student_a13.json`
* Checkpoint: `models/checkpoints/distilled_student_swahili.pt`
