Hi Christoph,

I like this project idea. I think that framing the problem as anomaly detection, rather than directly detecting “smurfing”, is the right conceptual move given the absence of reliable labels.

The lack of ground truth is not an issue in itself, but it does limit what you can claim. You are not validating correctness in the supervised sense; instead, you are showing that certain players behave in ways statistically inconsistent with their rating group. That is a valid objective, as long as you are explicit about it. In other words, your conclusions should be phrased in terms of detectable deviations and systematic differences, not definitive identification of smurfing.

Your proposed validation strategy is appropriate for this setting. Using engine-based metrics, such as centipawn loss, provides an external signal that grounds your analysis in measurable terms and helps demonstrate that the players you identify differ meaningfully from their peers. The idea of injecting synthetic anomalies is very good. It provides a controlled setting where you know what an anomaly looks like and allows you to test whether your method can recover it.

I would structure your evaluation clearly around these ideas. First, show that the players you flag are statistically different from the rest of the population. Then, connect this difference to an external signal, such as engine evaluations. Finally, use your synthetic experiments to demonstrate that your method can detect clear, known deviations. This progression will make your argument rigorous and easier to defend.

On the modeling side, I would keep things relatively simple and compare a small number of approaches, e.g., Isolation Forest, One-Class SVM, or a clustering-based method. The goal is not to build the most complex model, but to show that your conclusions are consistent across reasonable choices. As such, I would not recommend using a neural network as the primary model for this project. In your setting, the main challenges are defining meaningful features and establishing a validation strategy in the absence of labels, not model capacity. A complex model will not resolve these issues and may make your results harder to interpret.

Accordingly, I would be mindful of how you construct your features. They need to be comparable across players, which typically means controlling for factors such as the number of games, time controls, and possibly opponent strength. Also, be careful when interpreting your results: players may appear anomalous for reasons unrelated to the behavior you are targeting, e.g., rapid improvement, inconsistent play, or mismatched pools.

I would consider neural networks as representation learners or as autoencoders. Used this way, a neural network becomes a tool to improve representation or provide an alternative anomaly signal.

If you have access to richer game- or move-level data, you could use a neural network to learn embeddings of player behavior. The idea is to map sequences of moves, openings, or game statistics into a fixed-dimensional vector that captures patterns of play. Similar players (in terms of style or strength) should produce similar embeddings. You would then use these learned representations as input to your anomaly detection method, e.g., Isolation Forest or clustering, as mentioned above. This separates the problem into two parts: learning a good representation and detecting deviations in that space.

An alternative is to use an autoencoder. In this approach, you train a neural network to reconstruct typical player behavior from your feature set. The model learns a compressed internal representation of what is “normal”. At inference time, players whose behavior cannot be reconstructed well (i.e., high reconstruction error) are flagged as anomalous. This yields a natural anomaly score without requiring labels.

If you take either approach, there are a few important points to consider:

  *
The quality of the input representation is critical. For embeddings, this means carefully deciding what constitutes a “behavioral unit,” e.g., per game, per sequence of moves, or aggregated statistics. For autoencoders, this means ensuring that features are well-defined and comparable across players.
  *
Control for confounding factors such as time controls, number of games, and opponent strength. Otherwise, the model may learn differences unrelated to the phenomenon you are interested in.
  *
Keep the architecture simple. A small feedforward network is sufficient; there is no need for deep or highly tuned models.
  *
Most importantly, compare with simpler methods. If a neural approach does not yield clearer or more consistent results than classical anomaly detection, that is itself an important finding.

Hope it helps, I think there is great potential here.

Best,
Matteo