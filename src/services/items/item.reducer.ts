import {AnyAction} from 'redux';
import {ItemTypes} from './item.types';

export type itemlistType = {
  itemList: null;
  isLoading: boolean;
  error: null;
  itemA: null;
};

const initialState: itemlistType = {
  itemList: null,
  itemA: null,
  isLoading: false,
  error: null,
};
const itemReducer = (state = initialState, action: AnyAction) => {
  switch (action.type) {
    case ItemTypes.GET_ITEM_LIST_START:
    case ItemTypes.REMOVE_ITEM_START:
      return {...state, isLoading: true};

    case ItemTypes.GET_ITEM_LIST_SUCCESS:
      return {...state, isLoading: false, itemList: action.payload};
    case ItemTypes.ADD_ITEM_SUCCESS:
      return {
        ...state,
        itemList: state.itemList.concat(action.payload), // new item
        itemA: action.payload,
        isLoading: false,
      };
    case ItemTypes.REMOVE_ITEM_SUCCESS:
      return {
        ...state,
        itemList: state.itemList.filter(
          item => item._id !== action.payload._id,
        ),
        isLoading: false,
      };

    case ItemTypes.GET_ITEM_LIST_FAIL:
    case ItemTypes.ADD_ITEM_FAIL:
      return {...state, isLoading: false, error: action.payload};
    case ItemTypes.REMOVE_ITEM_FAIL:
      return {...state, isLoading: false, error: action.payload};
    default:
      return state;
  }
};

export default itemReducer;
