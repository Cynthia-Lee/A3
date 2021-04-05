# Dataset

# Header
# Consists of fields such as <From>, <Subject>, <Organization> and <Lines> fields.
# The <lines> field includes the number of lines in the document body

# Body
# The main body of the document. This is where you should extract features from.

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html?fbclid=IwAR2e_uoplSxSBWON3XZ69JA1Fnck-SFFE42PUKAVPi_quhe8CQk4qUnReWQ

categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
train_folder = ".\Selected 20NewsGroup\Training"

# data = load_files(train_folder)

# for root, dirs, files in os.walk(train_folder, topdown=False):
#     file_names = []
#     file_paths = []
#     for name in files:
#         path = (os.path.join(root, name))
#         file_paths.append(path)
#         file_names.append(name)
#     # print(file_paths, file_names)

# files = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
# print(files)