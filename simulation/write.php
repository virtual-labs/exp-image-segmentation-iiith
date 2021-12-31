<?php 

session_start();

if(isset($_POST["item"])) {
$item='<div class="item">';
$item.=$_POST["item"];
$item.='</div>';
}
writethis($item);
$_SESSION["currentS"]=$_SESSION["state"];

function writethis($item)
{$item=preg_replace( '/140/','281', $item);
$myfile=fopen("temp/".$_SESSION["id"]."p.html",'a');
	fwrite($myfile,	preg_replace( '/\\\"/','"', $item));
	fclose($myfile); 
}

?>
