import styled from 'styled-components/native';

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

//style={{height: 60, width: 50, marginLeft: 10, borderRadius: 12}}

export const ItemDescription = styled.View`
  flex-direction: row;
  max-width:35%;
  min-width: 35%;
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

export const RemoveItemNavigate = styled.View`
  flex-direction: row;
  max-width: 11%;
  min-width: 11%;
  justify-content: center;
  align-items: center;
  border-top-width: 1px;
  border-top-color: #8ECFAD;
  border-right-width: 1px;
  border-right-color: #8ECFAD;
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
`;
export const ItemNavigate = styled.View`
  flex-direction: row;
  max-width: 9%;
  min-width: 9%;
  justify-content: center;
  align-items: center;
  border-top-width: 1px;
  border-top-color: #8ECFAD;
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
`;

export const DeleteIcon = styled.Image`
  width: 20px;
  height: 26px;
`;

export const NoItem = styled.Text`
  color: ${props => props.theme.colors.text.dark};
  font-size: ${props => props.theme.fontSizes.caption};
  font-weight: ${props => props.theme.fontWeights.bold};
  margin-bottom: ${props => props.theme.space[3]};
  align-self: center;
`;