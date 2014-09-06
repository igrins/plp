var myApp = angular.module('myApp', []);

myApp.config(function($interpolateProvider) {
  $interpolateProvider.startSymbol('{[{');
  $interpolateProvider.endSymbol('}]}');
});

function MyCtrl($scope) {
  $scope.order_min = order_minmax[0];
  $scope.order_max = order_minmax[1];

  $scope.order = order_minmax[0];
  $scope.name = name;

  $scope.go = function() {
      if ($scope.order < order_minmax[0]) {
	  $scope.order = order_minmax[0];
      };
      if ($scope.order > order_minmax[1]) {
	  $scope.order = order_minmax[1];
      };
      myonclick($scope.order);
  }

  $scope.next_order = function() {
      $scope.order += 1;
      $scope.go();

  };

  $scope.prev_order = function() {
      $scope.order -= 1;
      $scope.go();
  };

  $scope.show_all_order = function() {
      myshowall();
  };
}
