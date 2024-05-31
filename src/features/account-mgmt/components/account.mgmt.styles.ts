import styled from 'styled-components/native';

export const AddItemContainer = styled.View`
  width: 100%;
  height: 100%;
  justify-items: center;
  padding-left: ${props => props.theme.space[6]}
  padding-right: ${props => props.theme.space[6]}
  background: rgba(255, 255, 255, 0.7);
`;
export const ItemTitleContainer = styled.View`
  flex-direction: row;
  justify-content: space-between
  padding-top: ${props => props.theme.space[3]}
  padding-bottom: ${props => props.theme.space[3]}
`;

export const SectionDivider = styled.View`
  border-bottom-width: 1px;
  border-bottom-color: ${props => props.theme.colors.ui.quinary};
  width: 100%;
`;

export const TierCollector = styled.View`
  padding-top: ${props => props.theme.space[4]};
`;
export const TierHeader = styled.View`
  flex-direction: row;
  justify-content: space-between;
`;
export const TierBoxLeft = styled.View`
  flex-direction: row;
  align-items: center;
`;
export const TierLevelIcon = styled.View`
  width: 30px;
  height: 30px;
`;
export const TierLevelText = styled.Text`
  font-weight: 500;
  position: absolute;
  left: 0;
  right: 0;
  top: 5px;
  bottom: 0;
  text-align: center;
  z-index: 1;
  color: #096c5c;
  font-size: 18px;
`;
export const TierLevelBlueText = styled.Text`
  font-weight: 500;
  position: absolute;
  left: 0;
  right: 0;
  top: 5px;
  bottom: 0;
  text-align: center;
  z-index: 1;
  color: #81bdca;
  font-size: 18px;
`;
export const TierLevelPurpleText = styled.Text`
  font-weight: 500;
  position: absolute;
  left: 0;
  right: 0;
  top: 5px;
  bottom: 0;
  text-align: center;
  z-index: 1;
  color: #8c8ae3;
  font-size: 18px;
`;

export const TierImage = styled.Image`
  width: 30px;
  height: 30px;
`;
export const TierLeftText = styled.Text`
  background-color: #82BDCA;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
`;
export const TierLeftGreenText = styled.Text`
  background-color: ${props => props.theme.colors.bg.secondary}
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
`;

export const TierLeftPurpleText = styled.Text`
  background-color: #8C8AE3;
  padding-left: ${props => props.theme.space[2]}
  padding-right: ${props => props.theme.space[2]}
  color: ${props => props.theme.colors.text.inverse};
`;

export const CurrentButton = styled.View`
  align-items: center;
  border-radius: 4px;
  background-color: ${props => props.theme.colors.bg.quinary};
  padding-top: ${props => props.theme.space[2]};
  padding-bottom: ${props => props.theme.space[2]};
  padding-left: ${props => props.theme.space[5]};
  padding-right: ${props => props.theme.space[5]};
`;
export const SubscribeButtonBlue = styled.View`
  align-items: center;
  border-radius: 4px;
  background-color: #35aaff;
  padding-top: ${props => props.theme.space[2]};
  padding-bottom: ${props => props.theme.space[2]};
  padding-left: ${props => props.theme.space[5]};
  padding-right: ${props => props.theme.space[5]};
`;
export const SubscribeButtonPink = styled.View`
  align-items: center;
  border-radius: 4px;
  background-color: #bf56ff;
  padding-top: ${props => props.theme.space[2]};
  padding-bottom: ${props => props.theme.space[2]};
  padding-left: ${props => props.theme.space[5]};
  padding-right: ${props => props.theme.space[5]};
`;
export const CurrentButtonText = styled.Text`
  color: ${props => props.theme.colors.text.inverse};
  font-size: ${props => props.theme.fontSizes.text};
  font-weight: ${props => props.theme.fontWeights.bold};
`;

export const TierCollectorList = styled.View`
  padding-top: ${props => props.theme.space[3]};
`;
export const TierListItem = styled.View`
  flex-direction: row;
  align-items: center;
  padding-bottom: ${props => props.theme.space[1]};
`;

export const ItemImage = styled.Image`
  width: 24px;
  height: 24px;
`;
export const TierListItemText = styled.Text`
  font-size: ${props => props.theme.fontSizes.subText};
  color: ${props => props.theme.colors.text.primary};
  padding-left: ${props => props.theme.space[1]};
`;
