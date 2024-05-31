import styled from 'styled-components/native';

export const ItemList = styled.View`
  flex-direction: row;
`;

export const ItemTitle = styled.View`
  flex-direction: row;
  max-width:45%;
  min-width: 45%;
  justify-content: space-between;
  align-items: center;
  border-top-width: 1px;
  border-right-width: 1px;
  border-top-color: #8ecfad;
  border-right-color: #8ecfad;
  padding-right: ${props => props.theme.space[2]}
  padding-top: ${props => props.theme.space[3]};
  padding-bottom: ${props => props.theme.space[3]};
  flex-grow: 1;
`;
export const ItemImage = styled.Image`
  width: 56px;
  height: 64px;
  border-radius: 8px;
`;
//style={{height: 60, width: 50, marginLeft: 10, borderRadius: 12}}
export const ItemTitleText = styled.Text`
  color: ${props => props.theme.colors.text.tertiary}
  padding-left: ${props => props.theme.space[2]}
  flex: 1
  font-weight: ${props => props.theme.fontWeights.bold}
`;
export const ItemDescription = styled.View`
  flex-direction: row;
  max-width:45%;
  min-width: 45%;
  justify-content: center;
  align-items: center;
  border-top-width: 1px;
  border-right-width: 1px;
  border-top-color: #8ECFAD;
  border-right-color: #8ECFAD;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
`;
export const ItemDescriptionText = styled.Text`
  color: ${props => props.theme.colors.text.primary};
  font-size: ${props => props.theme.fontSizes.subCaption};
  font-weight: ${props => props.theme.fontWeights.bold};
`;
export const ItemNavigate = styled.View`
  flex-direction: row;
  max-width: ${props => (props.screenName === 'RemoveItem' ? 10 : 10)}%;
  min-width: ${props => (props.screenName === 'RemoveItem' ? 10 : 10)}%;
  justify-content: center;
  align-items: center;
  border-top-width: 1px;
  border-top-color: #8ECFAD;
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
`;
export const OnTouch = styled.TouchableOpacity`
  width: 100%;
  padding-top: 50%;
  padding-bottom: 50%;
  align-items: center;
`;
export const ItemDetailImage = styled.Image`
  width: 7px;
  height: 12px;
`;
