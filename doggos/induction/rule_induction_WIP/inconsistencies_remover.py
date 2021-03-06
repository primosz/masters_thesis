import pandas as pd


class InconsistenciesRemover(object):

    def __init__(self, features_table, feature_labels):
        self.features_table = features_table
        self.feature_labels = feature_labels
        self.changed_decisions = 0

    def getOccurenceOfRows(self, df, remove_columns):
        if remove_columns:
            df = df.drop(remove_columns, axis=1, inplace=False)

        df = df.groupby(
            df.columns.tolist(),
            as_index=False).size().reset_index()

        df.rename(columns={'size': 'Occurence'}, inplace=True)

        return df

    def getOccurenceOfRowsWithoutRemove(self, df):
        df = df.groupby(
            df.columns.tolist(),
            as_index=False).size().reset_index()

        df.rename(columns={'size': 'Occurence'}, inplace=True)

        return df

    def getCertainDecisionRows(self, features_occurence, features_decisions_occurence):
        features_decision_numbers_ones = features_occurence.loc[
            features_occurence['Occurence'] == 1.].copy()

        for index, row in features_decision_numbers_ones.iterrows():
            for idx, row_with_decision in features_decisions_occurence.iterrows():
                if (row[self.feature_labels].values ==
                    row_with_decision[self.feature_labels].values).all():
                    features_decision_numbers_ones.at[
                        index, 'Decision'] = features_decisions_occurence.loc[
                        idx, 'Decision']

        return features_decision_numbers_ones.drop(['Occurence'],
                                                   axis=1,
                                                   inplace=False)

    def getNumberOfClearDecision(self, features_occurence, features_decisions_occurence):
        features_certain_decision = self.getCertainDecisionRows(
            features_occurence, features_decisions_occurence)

        if features_certain_decision.empty: return 0
        tmp_table = pd.merge(
            features_decisions_occurence,
            features_certain_decision,
            how='inner',
            on=self.feature_labels)

        if 'Decision_y' in tmp_table.columns:
            tmp_table = tmp_table.drop(['Decision_y'], axis=1).rename(
                index=str,
                columns={
                    "Decision_x": "Decision",
                    "Occurence_x": "Occurence"
                })

        number_of_clear_decision = pd.DataFrame(
            tmp_table.groupby(['Decision'],
                              as_index=False)['Occurence'].agg('sum'))

        return number_of_clear_decision

    def solveConflicts(self, number_of_conflicts_decision, problems_to_solve,
                       features_decisions_occurence, number_of_clear_decision, general_features_occurence):

        for _, row in number_of_conflicts_decision.iterrows():
            new_df = pd.DataFrame(columns={"Decision", "Probability"})

            for _, row_2 in problems_to_solve.iterrows():
                if (row[self.feature_labels].values == row_2[self.feature_labels]).all():

                    try:
                        occurence = (number_of_clear_decision.loc[
                            number_of_clear_decision['Decision'] == row_2[
                                ['Decision']].values[0]]).values[0][1]
                    except:
                        occurence = 0

                    probability = occurence / len(self.features_table)
                    new_df = new_df.append({
                        'Decision': row_2[['Decision']].values,
                        'Probability': probability
                    },
                        ignore_index=True)

            new_value = new_df.loc[new_df['Probability'].idxmax()]['Decision'][0]
            for idx, row_decision_table in features_decisions_occurence.iterrows():
                if (row[self.feature_labels].values == row_decision_table[self.feature_labels]).all():
                    if features_decisions_occurence.loc[features_decisions_occurence.index == idx].Decision.values[
                        0] != new_value:
                        # if self.settings.show_results:
                        # print("Current value: {}".format(features_decisions_occurence.loc[features_decisions_occurence.index == idx].Decision.values[0]))
                        # print("New value: {}".format(new_value))
                        # for idy, row_general_occurence in general_features_occurence.iterrows():
                        #     if (row_general_occurence[self.feature_labels].values == row_decision_table[self.feature_labels]).all():
                        #         if self.settings.show_results:
                        #             print(row_general_occurence.Occurence)
                        #         break
                        features_decisions_occurence.loc[idx, 'Decision'] = new_value
                        self.changed_decisions = self.changed_decisions + 1

        return features_decisions_occurence

    def inconsistenciesRemoving(self):
        features_decisions_occurence = self.getOccurenceOfRows(
            self.features_table, None)

        features_decisions_occurence = features_decisions_occurence.drop(['index'], axis=1)

        general_features_occurence = features_decisions_occurence.copy()
        self.samples = features_decisions_occurence.Occurence.sum()
        old_size = len(self.features_table)

        features_occurence = self.getOccurenceOfRows(self.features_table,
                                                     None)

        features_occurence = self.getOccurenceOfRows(self.features_table, ['Decision'])

        number_of_conflicts_decision = features_occurence[
            features_occurence.Occurence > 1]

        number_of_clear_decision = self.getNumberOfClearDecision(
            features_occurence, features_decisions_occurence)

        problems_to_solve = pd.merge(
            features_decisions_occurence,
            number_of_conflicts_decision,
            how='inner',
            on=self.feature_labels).drop(['Occurence_x', "Occurence_y"], axis=1)

        features_decisions_occurence = self.solveConflicts(
            number_of_conflicts_decision, problems_to_solve,
            features_decisions_occurence, number_of_clear_decision, general_features_occurence)
        decision_table = features_decisions_occurence.drop(['Occurence'],
                                                           axis=1).drop_duplicates(
            keep='first',
            inplace=False)
        # self.changed_decisions = old_size - len(features_decisions_occurence)

        return decision_table, self.changed_decisions
