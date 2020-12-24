import pandas as pd
import os
from classification_instance import ClassificationInstance
from sklearn.utils.testing import all_estimators

def all_but_one_evaluation(dataset_dir, clf=None):
	for sub_dir, label in [("methods", "BuggedMethods"), ("classes_no_aggregate", "Bugged")]:
		scores = []
		for e_type in ('one', 'all'):
			d_dir = os.path.join(dataset_dir, e_type, sub_dir)
			for data_type in os.listdir(d_dir):
				predict_one_type(clf, d_dir, data_type, e_type, label, scores)
		pd.DataFrame(scores).to_csv(os.path.join(dataset_dir, sub_dir + "_metrics.csv"), index=False, sep=';')


def predict_one_type(clf, d_dir, data_type, e_type, label, scores):
	d_type_dir = os.path.join(d_dir, data_type)
	training = pd.read_csv(os.path.join(d_type_dir, "training.csv"), delimiter=';').drop("Method_ids",
																						 axis=1,
																						 errors='ignore')
	testing = pd.read_csv(os.path.join(d_type_dir, "testing.csv"), delimiter=";")
	for col in testing:
		dt = testing[col].dtype
		if dt == int or dt == float:
			testing[col].fillna(0, inplace=True)
		else:
			testing[col].fillna(False, inplace=True)
	ci = ClassificationInstance(training, testing, None, d_type_dir, label=label, save_all=True)
	try:
		ci.predict(clf)
		ci_scores = dict(ci.scores)
		ci_scores.update({"type": e_type, "data_type": data_type})
		scores.append(ci_scores)
	except Exception as e:
		print(e)
		pass


def predict_one(training, testing, clf):
	training = pd.read_csv(training, delimiter=';')
	testing = pd.read_csv(testing, delimiter=';')
	ans = {}
	for col in testing:
		dt = testing[col].dtype
		if dt == int or dt == float:
			testing[col].fillna(0, inplace=True)
		else:
			testing[col].fillna(False, inplace=True)
	ci = ClassificationInstance(training, testing, None, None, save_all=False)
	try:
		ci.predict(clf)
		ans = dict(ci.scores)
	except:
		pass
	return ans


def predict_dir(data_dir, clf, proj_name):
	data = []
	classes_dir = os.path.join(data_dir, "classes")
	training, testing = list(map(lambda d: os.path.join(classes_dir, d), ['training.csv', 'testing.csv']))
	ans = predict_one(training, testing, clf)
	ans['data_type'] = "classes"
	ans['type'] = "classes"
	ans['proj_name'] = proj_name
	data.append(ans)
	scores = []
	for e_type in ('one', 'all'):
		d_dir = os.path.join(data_dir, e_type, 'classes')
		for data_type in os.listdir(d_dir):
			training, testing = list(map(lambda d: os.path.join(d_dir, data_type, d), ['training.csv', 'testing.csv']))
			ans = predict_one(training, testing, clf)
			ans['data_type'] = data_type
			ans['type'] = e_type
			ans['proj_name'] = proj_name
			data.append(ans)
		# pd.DataFrame(scores).to_csv(os.path.join(data_dir, "_metrics.csv"), index=False, sep=';')
	return data


def main():
	# clf = dict(all_estimators())["_BinaryGaussianProcessClassifierLaplace"](**{'copy_X_train': True, 'kernel': None, 'max_iter_predict': 100, 'n_restarts_optimizer': 0, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None, 'warm_start': False})
	clf = dict(all_estimators())["GaussianProcessClassifier"](**{'copy_X_train': True, 'kernel': None, 'max_iter_predict': 100, 'multi_class': 'one_vs_rest', 'n_jobs': None, 'n_restarts_optimizer': 0, 'optimizer': 'fmin_l_bfgs_b', 'random_state': None, 'warm_start': False})
	all_data = []
	data_dir = os.path.abspath(r"dataset")
	for x in filter(os.path.isdir, os.listdir(data_dir)):
		try:
			all_data.extend(predict_dir(os.path.join(data_dir, x), clf, x))
		except Exception as e:
			print(e)
			pass
	pd.DataFrame(all_data).to_csv(r"results.csv", index=False)


if __name__ == "__main__":
	main()