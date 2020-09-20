from cross_validate import train_test_split


def test_intersection():
    dummy = pd.DataFrame('other_col':np.arange(10,30,2),
                         'id':[1,1,2,2,3,3,4,4,5,5])
    
    train, test = train_test_split()
    
    set_train = set(train['id'])
    set_test = set(test['id'])
    
    assert len(set_train & set_test) == 0
    print("Case passed")

    
def test_col_id():
    dummy = pd.DataFrame('other_col':np.arange(10,30,2),
                         'Id':[1,1,2,2,3,3,4,4,5,5])
    try:
        train_test_split(dummy)
        assert False
    except ValueError as e:
        assert True
    print("Case passed")
    

if __name__ == "__main__":
    test_intersection()
    test_col_id()
    print("All test cases passed")