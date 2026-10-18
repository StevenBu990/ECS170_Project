from local_code.base_class.setting import setting

class Setting_Train_Test(setting):
    test_dataset = None
    
    def load_run_save_evaluate(self):
        training_data = self.dataset.load()
        testing_data = self.test_dataset.load()

        self.method.data = {'train': training_data, 'test': testing_data}

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()

        