# Data efficient landmark localization using memory augmented neural networks
Repository containing Code for Masters Thesis on "Data efficient landmark localization using memory augmented neural networks"

## Abstract
Biomedical image analysis frequently faces the issue of few available samples due to expensive or difficult data collection procedures. While deep learning often focuses on working with large datasets, for automated analysis of biomedical images, methods are required which are able to learn from few samples. In this work, iterative learning of landmarks with a single U-Net is proposed. The iterative task structure includes implicit memory as the prediction from the previous step is used in the next. This work shows that the thereby included recurrent inductive bias can improve landmark localization. In a second step, an explicit memory module in form of a neural turing machine gate is added and compared to attention gates using the same model architecture.
By extending a baseline U-Net with a neural turing machine gate or attention gate, localization error can be further decreased both on small datasets with less than 500 training samples and on an even smaller subset of less than 50 samples. 

