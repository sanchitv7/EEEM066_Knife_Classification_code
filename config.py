class DefaultConfigs(object):
    def __init__(self, n_classes, img_weight, img_height, batch_size, epochs, learning_rate, weight_decay):
        self.n_classes = n_classes  # number of classes
        self.img_weight = img_weight  # image width
        self.img_height = img_height  # image height
        self.batch_size = batch_size  # batch size
        self.epochs = epochs  # epochs
        self.learning_rate = learning_rate  # learning rate
        self.weight_decay = weight_decay


config = DefaultConfigs(n_classes=192,
                        img_weight=224,
                        img_height=224,
                        batch_size=32,
                        epochs=10,
                        learning_rate=0.005, 
                        weight_decay = 0.01)
